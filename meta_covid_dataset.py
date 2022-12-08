"""Dataloading for Long Covid in Meta Learning Framework."""
import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import dataset, sampler, dataloader
from sklearn.preprocessing import normalize
NUM_TRAIN_CLASSES = 1100
NUM_VAL_CLASSES = 100
NUM_TEST_CLASSES = 423
NUM_SAMPLES_PER_CLASS = 2

class LongCovidMetaDataset(dataset.Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './data'

    def __init__(self, num_support, num_query, data_filename):
        """Inits meta_covid_dataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # if necessary, download the Omniglot dataset
        if not os.path.isdir(self._BASE_PATH):
            assert("./data not available")

        self._data = pd.read_csv(os.path.join(_BASE_PATH, data_filename))

        self._data.iloc[:, :-1] = normalize(self._data.iloc[:,:-1], axis = 0, norm = 'max')

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        x_support, x_query = [], []
        y_support, y_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them

            class_sample = self._data[self._data['LongCovid'] == class_idx].sample(n = (self._num_support + self._num_query), replace = False)
            

            # split sampled examples into support and query
            x_support.extend(class_sample[:self._num_support, :-1])
            x_query.extend(class_sample[self._num_support:, :-1])
            y_support.extend([label] * self._num_support)
            y_query.extend([label] * self._num_query)

        # aggregate into tensors
        x_support = torch.stack(x_support)  # shape (N*S, C, H, W)
        y_support = torch.tensor(y_support)  # shape (N*S)
        x_query = torch.stack(x_query)
        y_query = torch.tensor(y_query)

        return x_support, y_support, x_query, y_query


class LongCovidSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = [0,1]
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_longcov_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == 'train':
        data_filename = "proteins_longcovid_target_metatrain.csv"
    elif split == 'val':
        data_filename = "proteins_longcovid_target_metaval.csv"
    elif split == 'test':
        data_filename = "proteins_longcovid_target_test.csv"
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=LongCovidMetaDataset(num_support, num_query, data_filename),
        batch_size=batch_size,
        sampler=LongCovidSampler(num_way, num_tasks_per_epoch),
        num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
