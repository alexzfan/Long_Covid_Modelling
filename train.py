"""
Train a model on the Hateful Memes Dataset
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import util

from args import get_train_args
from models import baseline_ff
from util import LongCovidDataset
from collections import OrderedDict
from sklearn import metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
from json import dumps
from sklearn.utils.class_weight import compute_class_weight
import os
import sys

def main(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set up logger and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training = True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()

    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # set seed
    log.info(f'Seed: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get Model
    log.info("Making model....")
    if(args.model_type == "baseline"):
        model = baseline_ff(hidden_size=args.hidden_size)
    else:
        raise Exception("Model provided not valid")

    model = nn.DataParallel(model, args.gpu_ids)

    # load the step if restarting
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0    

    # send model to dev and start training
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get checkpoint saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints = args.max_checkpoints,
                                 metric_name = args.metric_name,
                                 maximize_metric = args.maximize_metric,
                                 log = log)
    optimizer = optim.Adam(model.parameters(),
                            lr = args.lr,
                            betas = (0.9, 0.999),
                            eps = 1e-7,
                            weight_decay = args.l2_wd)
    

    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # load in data
    log.info("Building dataset....")
    if(args.model_type == "baseline"):
        dataset = LongCovidDataset(args.train_eval_file)

        # split the dataset to train and val
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        if args.shuffle_dataset:
            np.random.seed(args.seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                                                sampler=train_sampler)
        dev_loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                                        sampler=valid_sampler)
    else:
        raise Exception("Dataset provided not valid")
    # Start training
    log.info("Training...")
    steps_till_eval = args.eval_steps
    epoch = step // int(len(dataset) * (0.8))

    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}....')
        with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
            for x, y in train_loader:
                # forward pass here
                x = x.float().to(device)

                batch_size = args.batch_size
                optimizer.zero_grad()

                if(args.model_type == "baseline"):
                    score = model(x)
                else:
                    raise Exception("Model Type Invalid")

                # calc loss
                y = y.float().to(device)

                # weight the BCE
                weights = compute_class_weight(class_weight='balanced', classes= np.unique(y), y= y)
                weights=torch.tensor(weights,dtype=torch.float)
                criterion = nn.BCEWithLogitsLoss(reduction= 'none')

                loss = criterion(score, y.unsqueeze(1))
                for i in range(len(loss)):
                    if y[i] == 0:
                        loss[i] *= weights[0]
                    else:
                        loss[i] *= weights[1]
                loss = torch.mean(loss)
                loss_val = loss.item()

                # backward pass here
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step//batch_size)
                ema(model, step//batch_size)

                # log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch = epoch,
                                         loss = loss_val)
                tbx.add_scalar("train/loss", loss_val, step)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]['lr'],
                               step)
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Eval and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(args, 
                                                  model, 
                                                  dev_loader, 
                                                  device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)
                    
                    results_str = ", ".join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # tensorboard
                    log.info("Visualizing in TensorBoard")
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    
def evaluate(args, model, data_loader, device):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {} # id, prob and prediction
    full_score = []
    full_labels = []
    predictions = []

    acc = 0
    num_corrects, num_samples = 0, 0
    
    with torch.no_grad(), \
        tqdm(total=len(data_loader.dataset)) as progress_bar:
        for x, y in data_loader:
            # forward pass here
            x = x.float().to(device)
            # text = text.to(device)

            batch_size = args.batch_size

            if(args.model_type == "baseline"):
                score = model(x)
            else:
                raise Exception("Model Type Invalid")

            # calc loss
            y = y.float().to(device)
            weights = compute_class_weight(class_weight='balanced', classes = np.unique(y), y = y.numpy())
            weights=torch.tensor(weights,dtype=torch.float)
            criterion = nn.BCEWithLogitsLoss(reduction = 'none')
            
            preds, num_correct, acc = util.binary_acc(score, y.unsqueeze(1))
            loss = criterion(score, y.unsqueeze(1))
            for i in range(len(loss)):
                if y[i] == 0:
                    loss[i] *= weights[0]
                else:
                    loss[i] *= weights[1]
            
            loss_val = torch.mean(loss)
            nll_meter.update(loss_val.item(), batch_size)

            # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)
            predictions.extend(preds)
            full_score.extend(torch.sigmoid(score).tolist())
            full_labels.extend(y)


        acc = float(num_corrects) / num_samples

        # ROC
        y_score = np.asarray(full_score)
        y = np.asarray(full_labels).astype(int)
        print(predictions)
        auc = metrics.roc_auc_score(y, y_score)

    model.train()

    results_list = [("NLL", nll_meter.avg),
                    ("Acc", acc), 
                    ("AUROC", auc)]
    results = OrderedDict(results_list)
    
    return results, pred_dict



    
if __name__ == '__main__':
    main(get_train_args())