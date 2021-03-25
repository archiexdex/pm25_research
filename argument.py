
def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("--no",           type=int,                 help="the index of training process")
    parser.add_argument("--model",        type=str,                 help="model name")
    parser.add_argument("--seed",         type=int,   default=9487, help="seed for random number")
    parser.add_argument('--display_freq', type=int,   default=10,   help='display frequency for training')
    parser.add_argument('--gamma',        type=float, default=0.99, help='constant parameter')
    parser.add_argument('--batch_size',   type=int,   default=1024, help='batch size')
    parser.add_argument('--lr',           type=float, default=1e-3, help='')
    parser.add_argument('--total_epoch',  type=int,   default=1000, help='')
    parser.add_argument('--patience',     type=int,   default=5,    help='')
    parser.add_argument('--skip_site',    action='store_true', help='')
    parser.add_argument('--ratio',        type=int,   default=2,    help='')

    parser.add_argument('--memory_size',     type=int,   default=6570, help='The size of histry data 8760 for one year, 6570 for 9 months, 4380 for 6 months, 2160 for 3 months, 730 for 1 month')
    parser.add_argument('--window_size',     type=int,   default=6,   help='The size of history data period, it only use in fudan model.')
    parser.add_argument('--source_size',     type=int,   default=24,   help='The size of current data period.')
    parser.add_argument('--target_size',     type=int,   default=8,   help='')
    parser.add_argument('--threshold',       type=int,   default=70,  help='')
    parser.add_argument('--shuffle',         type=int,   default=1,   help='')
    parser.add_argument('--is_transform',    type=int,   default=1,   help='')
    parser.add_argument('--use_ext',         action='store_true',   help='')

    parser.add_argument('--input_dim',       type=int,   default=16,  help='')
    parser.add_argument('--output_dim',      type=int,   default=1,   help='')
    parser.add_argument('--emb_dim',         type=int,   default=32,  help='')
    parser.add_argument('--hid_dim',         type=int,   default=32,  help='')
    parser.add_argument('--dropout',         type=float, default=0.6, help='')
    parser.add_argument('--bidirectional',   type=bool,  default=False,  help='')

    parser.add_argument('--origin_train_dir',       type=str,   default="data/origin/train",           help='')
    parser.add_argument('--origin_valid_dir',       type=str,   default="data/origin/valid",           help='')
    parser.add_argument('--norm_train_dir',         type=str,   default="data/norm/train",             help='')
    parser.add_argument('--norm_valid_dir',         type=str,   default="data/norm/valid",             help='')
    parser.add_argument('--thres_train_dir',        type=str,   default="data/thres/train",             help='')
    parser.add_argument('--thres_valid_dir',        type=str,   default="data/thres/valid",             help='')
    parser.add_argument('--ext_train_dir',          type=str,   default="data/ext/train",             help='')
    parser.add_argument('--ext_valid_dir',          type=str,   default="data/ext/valid",             help='')
    parser.add_argument('--nonext_train_dir',       type=str,   default="data/nonext/train",             help='')
    parser.add_argument('--nonext_valid_dir',       type=str,   default="data/nonext/valid",             help='')
    parser.add_argument('--mean_path',              type=str,   default="data/train_mean.json",        help='')
    parser.add_argument('--std_path',               type=str,   default="data/train_std.json",         help='')
    parser.add_argument('--threshold_path',         type=str,   default="data/train_threshold.json",   help='')
    parser.add_argument('--cpt_dir',                type=str,   default="checkpoints",                 help='')
    parser.add_argument('--log_dir',                type=str,   default="logs",                        help='')
    parser.add_argument('--test_results_dir',       type=str,   default="test_results",                help='')
    parser.add_argument('--visual_results_dir',     type=str,   default="visual_results",              help='')
    parser.add_argument('--config_dir',             type=str,   default="configs",                     help='')
    
    
    
    
    return parser
