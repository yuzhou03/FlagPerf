# chip vendor, required
vendor: str = None
"""model params"""
name: str = "gcn"

data_dir = "/raid/dataset/gcn/cora/"
# random seed
seed: int = 42

do_train = True
fp16 = False
dist_backend: str = None
train_batch_size: int = None
eval_batch_size: int = None

# required params

# local_rank for distributed training on gpus
local_rank: int = 0
# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 10

gradient_accumulation_steps:int = 1


# Validate during training pass.
fastmode: bool = True

# Number of hidden units.
hidden: int = 16
# Dropout rate (1 - keep probability).
dropout: float = 0.5

# optimizer params
# Initial learning rate.
lr: float = 0.01
# Weight decay (L2 loss on parameters).
weight_decay: float = 5e-4

# Number of epochs to train.
max_epochs: int = 400

# Disables CUDA training.
no_cuda: bool = False

# target accurracy
target_acc: float = 0.815

cuda: bool = None