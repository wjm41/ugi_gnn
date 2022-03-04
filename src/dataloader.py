import pandas as pd


class UltraLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, pkl_path, inds, y_scaler, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param: csv_path (string): path to .csv containing molecular data
        :param: inds (array of ints): train/test indices to select subset of data
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        self.df = pd.read_pickle(pkl_path).reset_index().iloc[inds]
        self.df.reset_index(drop=True, inplace=True)
        self.y_scaler = y_scaler
        self.dataset_len = len(self.df)

        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(
                frac=1, random_state=42).reset_index(drop=True)

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        df_batch = self.df.iloc[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        # return batch
        return df_batch['smiles'].values, self.y_scaler.transform(df_batch['dockscore'].to_numpy().reshape(-1, 1))

    def __len__(self):
        return self.n_batches
