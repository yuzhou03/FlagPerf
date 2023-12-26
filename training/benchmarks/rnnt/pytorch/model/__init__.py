
from .model_rnnt import RNNT
from model import config
from utils.mlperf import logging


def create_model(args, tokenizer, rnnt_config):
    # set up the model
    rnnt_config = config.rnnt(args)
    logging.log_event(logging.constants.MODEL_WEIGHTS_INITIALIZATION_SCALE, value=args.weights_init_scale)
    
    if args.weights_init_scale is not None:
        rnnt_config['weights_init_scale'] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale

    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    return model