import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import human_len


class BigSmilesLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self,
                 file_path: str = None,
                 df: pd.DataFrame = None,
                 inds: np.ndarray = None,
                 y_col: str = 'dockscore',
                 scale_y: bool = False,
                 y_scaler: StandardScaler = None,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 random_state: bool = None,
                 weights=None):

        if df is not None:
            self.df = df
        elif file_path is not None:
            if file_path.split('.')[-1] == 'pkl':
                self.df = pd.read_pickle(file_path).reset_index()
            elif file_path.split('.')[-1] == 'csv':
                self.df = pd.read_csv(file_path).reset_index()
        else:
            raise ValueError(
                'Either df or file_path have to be specified to create a dataloader!')

        if inds is not None:
            self.df = self.df.iloc[inds]
        self.df = self.df.reset_index(drop=True)
        self.dataset_len = len(self.df)
        self.random_state = random_state
        self.y_col = y_col

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

        # initialise scaling
        if scale_y:
            self.y_scaler = StandardScaler()

            self.df[y_col] = self.y_scaler.fit_trainsform(
                self.df[y_col].to_numpy().reshape(-1, 1))

        # apply external scaling
        elif y_scaler is not None:
            self.df[y_col] = y_scaler.transform(
                self.df[y_col].to_numpy().reshape(-1, 1))

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(
                frac=1, random_state=self.random_state).reset_index(drop=True)

        if self.weights is not None:
            self.df = self.df.sample(
                frac=1, random_state=self.random_state, replace=True, weights=self.weights).reset_index(drop=True)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        df_batch = self.df.iloc[self.i:self.i+self.batch_size]
        self.i += self.batch_size

        return df_batch['smiles'].values, df_batch[self.y_col]

    def __len__(self):
        return self.dataset_len


def load_data(args):
    if args.train_path.split('.')[-1] == 'pkl':
        train_df = pd.read_pickle(args.train_path).reset_index()
    elif args.train_path.split('.')[-1] == 'csv':
        train_df = pd.read_csv(args.train_path).reset_index()
    len_train = len(train_df)

    if args.val_path is not None:  # external validation set
        train_loader = BigSmilesLoader(
            df=train_df, inds=None, batch_size=args.batch_size, shuffle=True, scale_y=True)
        val_loader = BigSmilesLoader(
            args.val_path, inds=None, batch_size=args.batch_size, shuffle=False, y_scaler=train_loader.y_scaler)

    elif args.val:  # random train/val split
        val_ind = np.random.choice(
            np.arange(len_train), args.val_size, replace=False)
        train_ind = np.delete(np.arange(len_train), val_ind)

        train_loader = BigSmilesLoader(
            df=train_df, inds=train_ind, batch_size=args.batch_size, shuffle=True, scale_y=True)
        val_loader = BigSmilesLoader(
            df=train_df, inds=val_ind, batch_size=args.batch_size, shuffle=False, y_scaler=train_loader.y_scaler)
    else:  # no validation set
        train_loader = BigSmilesLoader(
            df=train_df, inds=None, batch_size=args.batch_size, shuffle=True, scale_y=True)
        val_loader = None

    logging.info(f'Length of dataset: {human_len(len_train)}')
    logging.info(f'Length of training set: {human_len(train_loader)}')

    logging.info(f'Number of epochs to train: {human_len(args.n_epochs)}')
    logging.info(
        f'Number of batches (size {args.batch_size}) per epoch: {human_len(train_loader.n_batches)}')

    if val_loader is not None:
        logging.info(f'Length of validation set: {human_len(val_loader)}')
        return train_loader, val_loader
    else:
        return train_loader
