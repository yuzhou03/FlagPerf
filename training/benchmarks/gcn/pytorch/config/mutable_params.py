"""mutable_params defines parameters that can be replaced by vendor"""
mutable_params = [
    "dist_backend",
    "train_batch_size",
    "eval_batch_size",
    "lr",
    "weight_decay",
    "seed",
    "vendor",
]

mutable_params += ["local_rank", "do_train", "data_dir"]