# required parameters
vendor: str = None
# random seed
seed: int = 1
name: str = "rnnt"
# torch.backends.cudnn.benchmark
cudnn_benchmark: bool = True
# torch.backends.cudnn.deterministic
cudnn_deterministic: bool = True

"""Training parameters"""
# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 1
# Number of epochs for the entire training
max_epochs: int = 100
# Initial epochs of increasing learning rate
warmup_epochs: int = 6

# disable uniform initialization of batchnorm layer weight
disable_uniform_initialize_bn_weight: bool = False
""" lr_scheduler parameters"""
# peak learning rate
lr: float = 4e-3
# minimum learning rate
min_lr: float = 1e-5
"""optimizer parameters"""
# gamma factor for exponential lr scheduler
lr_exp_gamma: float = 0.935
# weight decay for the optimizer
weight_decay: float = 1e-3
# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 1
# If enabled, gradient norms will be logged
log_norm: bool = False
# If provided, gradients will be clipped above this norm
clip_norm: float = 1
# Beta 1 for optimizer
beta1: float = 0.9
# Beta 2 for optimizer
beta2: float = 0.999
# Discount factor for exp averaging of model weights
ema: float = 0.999
"""Precision parameters"""
# Use mixed precision training
amp: bool = False
# Static loss scaling, positive power of 2 values can improve fp16 convergence. If None, dynamicloss scaling is used.
loss_scale: float = 4096
fp16: bool = False
# Window over which to raise/lower dynamic scale
loss_scale_window: float = 1000
# Clip threshold for gradients
grad_clip_thresh: float = 1.0
# Minimum loss scale for dynamic loss scale
min_scale: float = 1
"""distributed parameters"""
# load and save args
# Path to a directory containing a model checkpoint.
init_checkpoint: str = None
"""data parameters"""
# Training data dir
data_dir: str = None
# Number of workers to use for dataloading
num_workers: int = 1

# Effective batch size per GPU (might require grad accumulation)
train_batch_size: int = 128
# Evaluation time batch size
eval_batch_size: int = 128

val_batch_size: int = 1

# trainer args
do_train: bool = True

# target WER(Word Error Rate) accuracy for training
target_wer: float = 0.058
"""Distributed parameters"""
distributed: bool = False
# Whether to read local rank from ENVVAR
use_env: bool = True
# local_rank for distributed training on gpus or other accelerators
local_rank: int = -1
# Communication backend for distributed training on gpus
dist_backend: str = "nccl"
# Distributed Data Parallel type
ddp_type: str = "native"
"""device parameters"""
device: str = None
n_device: int = 1
"""Dataset parameters"""

# feature and checkpointing parameters
# Use DALI pipeline for fast data processing. choices=['cpu', 'gpu']
dali_device: str = 'cpu'
# Try to resume from last saved checkpoint
resume: bool = False

# Path to a checkpoint for resuming training
ckpt: str = ''
# Saves model checkpoint at the end of training
save_at_the_end: bool = False
# Checkpoint saving frequency in epochs
save_frequency: int = 10
# Milestone checkpoints to keep from removing
keep_milestones: int = 0
# Epoch on which to begin tracking best checkpoint (dev WER)
save_best_from: int = 200
# Number of epochs between evaluations on dev set
val_frequency: int = 1
# Number of steps between printing training stats
log_frequency: int = 25
# Number of steps between printing sample decodings
prediction_frequency: int = None
# Path of the model configuration file
model_config: str = 'baseline_v3-1023sp.yaml'
# If provided, samples will be grouped by audio duration, to this number of buckets
# for each bucket, random samples are batched, and finally all batches are randomly shuffled
num_buckets: int = 6
# Paths of the training dataset manifest file
train_manifests: list = ["librispeech-train-clean-100-wav.json",  "librispeech-train-clean-360-wav.json" ,"librispeech-train-other-500-wav.json"]
# Paths of the evaluation datasets manifest files
val_manifests: list = ["librispeech-dev-clean-wav.json"]
# Discard samples longer than max_duration
max_duration: float = 0
# Path to save the training logfile.
log_file: str = None
# maximum number of symbols per sample can have during eval
max_symbol_per_sample: int = None