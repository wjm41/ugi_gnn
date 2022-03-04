import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import human_len


class PklLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self,
                 file_path: str,
                 inds: np.ndarray = None,
                 y_scaler: StandardScaler = None,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 random_state: bool = None):

        if file_path.split('.')[-1] == 'pkl':
            self.df = pd.read_pickle(file_path).reset_index()
        elif file_path.split('.')[-1] == 'csv':
            self.df = pd.read_csv(file_path).reset_index()

        if inds is not None:
            self.df = self.df.iloc[inds]
        self.df = self.df.reset_index(drop=True)
        self.dataset_len = len(self.df)
        self.random_state = random_state

        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

        # scale dockscore
        if y_scaler is not None:
            self.df['dockscore'] = y_scaler.transform(
                self.df['dockscore'].to_numpy().reshape(-1, 1))

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(
                frac=1, random_state=self.random_state).reset_index(drop=True)

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        df_batch = self.df.iloc[self.i:self.i+self.batch_size]
        self.i += self.batch_size

        return df_batch['smiles'].values, df_batch['dockscore']

    def __len__(self):
        return self.n_batches


def load_data(args, log=True):
    len_df = len(pd.read_pickle(args.path))

    y_scaler = StandardScaler()

    if args.val:
        val_ind = np.random.choice(
            np.arange(len_df), args.val_size, replace=False)
        train_ind = np.delete(np.arange(len_df), val_ind)
    else:
        train_ind = range(len_df)

    y_scaler = y_scaler.fit(pd.read_pickle(args.path)[
                            'dockscore'].to_numpy()[train_ind].reshape(-1, 1))

    train_loader = PklLoader(
        args.path, inds=train_ind, batch_size=args.batch_size, shuffle=True, y_scaler=y_scaler)
    val_loader = None

    if args.val_path is not None:
        val_loader = PklLoader(
            args.val_path, inds=None, batch_size=args.batch_size, shuffle=False, y_scaler=y_scaler)
    elif args.val:
        val_loader = PklLoader(
            args.path, inds=val_ind, batch_size=args.batch_size, shuffle=False, y_scaler=y_scaler)

    if log:
        print(
            f'Length of dataset: {len_df}({human_len(pd.read_pickle(args.path))})')
        print(f'Length of training set: {human_len(train_ind)}')
        print(f'Length of validation set: {human_len(val_ind)}')

        print(f'Number of epochs to train: {args.n_epochs}')
        print(
            f'Number of batches per epoch: {int(len(train_ind)/args.batch_size)}')
    if val_loader is not None:
        return {'train_loader': train_loader, 'val_loader': val_loader}
    else:
        return {'train_loader': train_loader}
