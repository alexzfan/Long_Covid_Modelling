"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
import higher

import meta_covid_dataset
import meta_util
import sys
import random

AUG_NET_DIM = 30000
INNER_NET_DIM = 500
KERNEL_SIZE = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600
RESNET_CHANNEL = 3
INNER_MODEL_SIZE = 4



class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_input_channels,
            num_outputs,
            num_inner_steps,
            aug_net_size,
            num_augs,
            aug_noise_prob,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            l2_wd,
            log_dir,
            debug
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
        """
        self.debug = debug
        self.num_input_channels = num_input_channels

        # construct feature extractor
        self._aug_net_size = aug_net_size
        self._aug_noise_prob = aug_noise_prob
        self._num_augs = num_augs

        self._aug_net = nn.Sequential()
        in_channel = self.num_input_channels
        for i in range(self._aug_net_size):
            if i == self._aug_net_size - 1:
                self._aug_net.append(meta_util.aug_net_block(in_channel, self.num_input_channels, self._aug_noise_prob, self._num_augs))
            else:
                self._aug_net.append(meta_util.aug_net_block(in_channel, AUG_NET_DIM, self._aug_noise_prob, self._num_augs))
                in_channel = AUG_NET_DIM
        self._aug_net = self._aug_net.to(DEVICE)


        # make inner model

        self._inner_net = nn.Sequential(
            nn.Linear(self.num_input_channels, INNER_NET_DIM),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(INNER_NET_DIM, INNER_NET_DIM),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(INNER_NET_DIM, INNER_NET_DIM),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(INNER_NET_DIM, num_outputs)
        ).to(DEVICE)
            
        self._num_inner_steps = num_inner_steps

        
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(
            list(self._aug_net.parameters())+
            list(self._inner_net.parameters()) ,
            lr=self._outer_lr,
            weight_decay = l2_wd
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0


    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            x_support, y_support, x_query, y_query = task
            x_support = x_support.to(DEVICE)
            y_support = y_support.to(DEVICE)
            x_query = x_query.to(DEVICE)
            y_query = y_query.to(DEVICE)

            # does the "augmentation"
            support_augs = torch.cat([x_support for _ in range(self._num_augs)], dim = 0)
            labels_augs = torch.cat([y_support for _ in range(self._num_augs)], dim = 0)

            support_augs = torch.unsqueeze(support_augs, 1)
            support_augs = self._aug_net(support_augs)
            
            # use higher
            inner_opt = torch.optim.SGD(self._inner_net.parameters(), lr=1e-1)

            with higher.innerloop_ctx(
                self._inner_net, inner_opt, copy_initial_weights=not train, track_higher_grads=train
            ) as (fnet, diffopt):

                # adapt in inner loop
                support_accs = []
                for _ in range(self._num_inner_steps):
                        spt_logits = fnet(support_augs)
                        spt_loss = F.cross_entropy(spt_logits, labels_augs)

                        support_accs.append(meta_util.score(spt_logits, labels_augs))
                        diffopt.step(spt_loss)
                spt_logits = fnet(support_augs)
                support_accs.append(meta_util.score(spt_logits, labels_augs))
                accuracies_support_batch.append(support_accs)

                # query time
                x_query = torch.unsqueeze(x_query, 1)
                qry_logits = fnet(x_query)
                qry_loss = F.cross_entropy(qry_logits, y_query)
                accuracy_query_batch.append(meta_util.score(qry_logits, y_query))

                if train:
                    qry_loss.backward()

                outer_loss_batch.append(qry_loss.detach())

                

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._aug_net.load_state_dict(state['aug_net'])
            self._inner_net.load_state_dict(state['inner_net'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(aug_net = self._aug_net.state_dict(),
                inner_net = self._inner_net.state_dict(),
                optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml/meta_covid.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    num_input_channels = 23074

    maml = MAML(
        num_input_channels,
        args.num_way,
        args.num_inner_steps,
        args.aug_net_size,
        args.num_augs,
        args.aug_noise_prob, 
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        args.l2_wd,
        log_dir,
        args.debug
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}, '
            f'num_augs={args.num_augs}'
        )

        dataloader_train = meta_covid_dataset.get_longcov_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = meta_covid_dataset.get_longcov_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )

        maml.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = meta_covid_dataset.get_longcov_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--aug_net_size', type=int, default=1,
                        help='how many conv layers in augmentation network')       
    parser.add_argument('--num_augs', type=int, default=1,
                        help='how many sets of augmentations')    
    parser.add_argument('--aug_noise_prob', type=float, default=0.1,
                        help='likelihood to inject noise in augmentation layer')                     
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--l2_wd', type=float, default=1e-4,
                        help='l2 weight decay for outer loop')            
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--debug', default=False, action = 'store_true',
                        help='debug by reducing to base maml')  

    main_args = parser.parse_args()
    main(main_args)
