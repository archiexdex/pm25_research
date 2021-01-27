def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("--no",           type=int,                 help="the index of training process")
    parser.add_argument("--seed",         type=int,                 help="seed for random number")
    parser.add_argument('--display_freq', type=int,   default=10,   help='display frequency for training')
    parser.add_argument('--gamma',        type=float, default=0.99, help='constant parameter')
    parser.add_argument('--batch_size',   type=int,   default=128,  help='batch size')
    parser.add_argument('--pos_weight',   type=float,   default=5/95,  help='weight for binary cross entropy')


    parser.add_argument('--origin_train_dir',       type=str,   default="dataset/origin/train",  help='')
    parser.add_argument('--origin_valid_dir',       type=str,   default="dataset/origin/valid",  help='')
    parser.add_argument('--cpt_dir',                type=str,   default="checkpoints",           help='')
    parser.add_argument('--log_dir',                type=str,   default="logs",                  help='')
    parser.add_argument('--test_results_dir',       type=str,   default="test_results",          help='')
    parser.add_argument('--visual_results_dir',     type=str,   default="visual_results",        help='')
    
    
    
    
    return parser
