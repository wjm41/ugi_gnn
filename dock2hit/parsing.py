
def add_io_args(parser):
    group_io = parser.add_argument_group("I/O")
    group_io.add_argument('-path_to_train_data', type=str, required=True,
                          help='Path to the data.csv file.')
    group_io.add_argument('-smiles_col', type=str, default='smiles',
                          help='the name of the column with the smiles of the dataset.')
    group_io.add_argument('-y_col', type=str, default='dockscore',
                          help='the name of the column with y values to-be-learned.')
    group_io.add_argument('-log_dir', type=str, default=None,
                          help='directory containing tensorboard logs')
    group_io.add_argument('-save_dir', type=str, default=None,
                          help='directory for saving model params. If None (default), no model saving will be done')
    group_io.add_argument('-path_to_load_checkpoint', default=None,
                          help='name for directory containing saved model params checkpoint file for continued training. If None (default), no model loading will occur.')
    return


def add_data_args(parser):
    group_data = parser.add_argument_group("Training - data")
    group_data.add_argument('-batch_size', type=int, default=32,
                            help='int specifying batch_size for training and evaluations')
    group_data.add_argument('-minibatch_size', type=int, default=32,
                            help='int specifying minibatch_size for training and evaluations')
    group_data.add_argument('-steps_per_log', type=int, default=100,
                            help='int specifying number of steps per validation and tensorboard log')
    group_data.add_argument('-batches_per_log', type=int, default=1000,
                            help='int specifying number of steps per validation and tensorboard log')
    group_data.add_argument('-save_batch', type=int, default=5000,
                            help='int specifying number of batches per model save.')
    group_data.add_argument('-n_epochs', type=int, default=1,
                            help='int specifying number of random epochs to train for.')
    group_data.add_argument('--test_set_size', type=float, default=0.1,
                            help='float in range [0, 1] specifying fraction of dataset to use as test set')
    group_data.add_argument('-random_train_val_split', action='store_true',
                            help='whether or not to do random train/val split and log val_loss')
    group_data.add_argument('-size_of_val_set', type=int, default=100,
                            help='Integer size of training set datapoints to use as random validation set.')
    group_data.add_argument('-path_to_external_val', type=str, default=None,
                            help='path to separate validation set ; if not None, overwrites -val options')
    return


def add_optim_args(parser):
    group_optim = parser.add_argument_group("Training - optimizer")
    group_optim.add_argument('-optimizer', type=str, default='Adam',
                             choices=['Adam', 'AdamHD', 'SGD',
                                      'SGDHD', 'FelixHD', 'FelixExpHD'],
                             help='name of optimizer to use during training.')
    group_optim.add_argument('-lr', type=float, default=1e-3,
                             help='float specifying learning rate used during training.')
    group_optim.add_argument('-hypergrad_lr', type=float, default=1e-3,
                             help='float specifying hypergradient learning rate used during training.')
    group_optim.add_argument('-hypergrad_lr_decay', type=float, default=1e-5,
                             help='float specifying hypergradient lr decay used during training.')
    group_optim.add_argument('-weight_decay', type=float, default=1e-4,
                             help='float specifying hypergradient weight decay used during training.')
    group_optim.add_argument('-hypergrad_warmup', type=int, default=100,
                             help='Number of steps warming up hypergrad before using for optimisation.')
    return

# TODO make these inplace
