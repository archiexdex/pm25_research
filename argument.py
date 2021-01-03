def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("--no",           type=int,                 help="the index of training process")
    parser.add_argument('--display_freq', type=int,   default=10,   help='display frequency for training')
    parser.add_argument('--gamma',        type=float, default=0.99, help='constant parameter')
    parser.add_argument('--batch_size',   type=int,   default=512,  help='batch size')

    parser.add_argument('--cpt_dir',      type=str,   default="checkpoints",  help='')
    parser.add_argument('--log_dir',      type=str,   default="logs",         help='')
    
    
    
    
    return parser
