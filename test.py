"""
Test a model on the Hateful Memes Dataset
"""
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_test_args
from models import baseline_ff
from util import LongCovidDataset
from collections import OrderedDict
from sklearn import metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
from json import dumps
import os
from os.path import join

def main(args):
    # Set up logging
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set up logger and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training = False)
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()

    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get Model
    log.info("Making model....")
    if(args.model_type == "baseline"):
        model = baseline_ff(hidden_size=args.hidden_size)
    else:
        raise Exception("Model provided not valid")

    model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')

    # get data loader
    if(args.model_type == "baseline"):
        test_dataset = LongCovidDataset(args.test_eval_file)                           
        test_loader = data.DataLoader(test_dataset,
                                    batch_size = args.batch_size,
                                    shuffle = True,
                                    num_workers = args.num_workers)
    else:
        raise Exception("Model provided not valid")

    log.info(f"Evaluating on {args.split} split...")
    nll_meter = util.AverageMeter()

    model = util.load_model(model, args.load_path, args.gpu_ids, return_step = False)
    model = model.to(device)
    model.eval()

    pred_dict = {} # id, prob and prediction
    full_score = []
    full_labels = []
    full_preds = []

    acc = 0
    num_corrects, num_samples = 0, 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad(), \
        tqdm(total=len(test_dataset)) as progress_bar:
        for x, labels in test_loader:
            # forward pass here
            x = x.float().to(device)
            # text = text.to(device)

            batch_size = args.batch_size

            if(args.model_type == "baseline"):
                score = model(x)
            else:
                raise Exception("Model Type Invalid")

            # calc loss
            labels = labels.float().to(device)
            preds, num_correct, acc = util.binary_acc(score, labels.unsqueeze(1))
            loss = criterion(score, labels.unsqueeze(1))
            nll_meter.update(loss.item(), batch_size)

            # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)

            full_preds.extend(preds)
            full_score.extend(torch.sigmoid(score).tolist())
            full_labels.extend(labels)

            # update 
            pred_dict.update(pred_dict_update)

        acc = float(num_corrects) / num_samples

        # ROC
        y_score = np.asarray(full_score)
        y = np.asarray(full_labels).astype(int)

        auc = metrics.roc_auc_score(y, y_score)
        df = pd.DataFrame(list(zip(full_score, full_preds, full_labels)), columns =['probs', 'preds', 'labels'])
        sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
        df.to_csv(sub_path, encoding = "utf-8")

        print("Acc: {}, AUROC: {}".format(acc, auc))



if __name__ == '__main__':
    main(get_test_args())
