# chip vendor, required
vendor: str = None

# random seed
seed: int = 1234
"""model args"""
name: str = "tacotron2"

"""Training parameters"""
# max epochs for training
max_epochs: int = 1501
# disable uniform initialization of batchnorm layer weight
disable_uniform_initialize_bn_weight: bool = False

""" lr_scheduler parameters"""
# initial learning rate
learning_rate: float = 0.1
# Epochs after which decrease learning rate
lr_anneal_steps: list = [500, 1000, 1500]
# Factor for annealing learning rate
lr_anneal_factor: float = 0.3

"""optimizer args"""
# weight decay coefficient for L2 regularization
weight_decay: float = 1e-6
# momentum for SGD optimizer
momentum: float = 0.9
"""Precision parameters"""
amp: bool = False
# Static loss scaling, positive power of 2 values can improve fp16 convergence. If None, dynamicloss scaling is used.
loss_scale: float = 4096
fp16: bool = True
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
num_workers: int = 2
# Total batch size for training.
train_batch_size: int = 128
# Total batch size for validating.
eval_batch_size: int = 128
# Maximum sequence length to process
seq_length: int = 200
# trainer args
do_train: bool = True

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: int = 1

# number of training samples to run a evaluation once
eval_interval_samples: int = 20000

# Sample to begin performing eval.
eval_iter_start_samples: int = 1

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 1

# target val_loss to converge for training
target_val_loss: float = 0.4852
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

disable_uniform_initialize_bn_weight: bool = False
# Enable cudnn
cudnn_enabled: bool = True
# Run cudnn benchmark
cudnn_benchmark: bool = False
"""Dataset parameters"""
# Loads mel spectrograms from disk instead of computing them on the fly
load_mel_from_disk: bool = True
# Path to training filelist
training_files: str = "filelists/ljs_mel_text_train_filelist.txt"
# Path to validation filelist
validation_files: str = "filelists/ljs_mel_text_val_filelist.txt"
# Type of text cleaners for input text
text_cleaners: list = ['english_cleaners']
"""Audio parameters"""
# Maximum audiowave value
max_wav_value: float = 32768.0
# Sampling rate
sampling_rate: int = 22050
# Filter length
filter_length: int = 1024
# Hop (stride) length
hop_length: int = 256
# Window length
win_length: int = 1024
# Minimum mel frequency
mel_fmin: float = 0.0
# Maximum mel frequency
mel_fmax: float = 8000
"""Misc parameters"""
# Number of bins in mel-spectrograms
n_mel_channels: int = 80
# Use mask padding
mask_padding: bool = False
"""Symbols parameters"""
# Number of symbols in dictionary
n_symbols: int = 148
# Input embedding dimension
symbols_embedding_dim: int = 512
"""Encoder parameters"""
# Encoder kernel size
encoder_kernel_size: int = 5
# Number of encoder convolutions
encoder_n_convolutions: int = 3
# Encoder embedding dimension
encoder_embedding_dim: int = 512
"""Decoder parameters"""
# Number of frames processed per step
n_frames_per_step: int = 1
# Number of units in decoder LSTM
decoder_rnn_dim: int = 1024
# Number of ReLU units in prenet layers
prenet_dim: int = 256
# Maximum number of output mel spectrograms
max_decoder_steps: int = 2000
# Probability threshold for stop token
gate_threshold: float = 0.5
# Dropout probability for attention LSTM
p_attention_dropout: float = 0.1
# Dropout probability for decoder LSTM
p_decoder_dropout: float = 0.1
# Stop decoding once all samples are finished
decoder_no_early_stopping: bool = False
"""Mel-post processing network parameters"""
# Postnet embedding dimension
postnet_embedding_dim: int = 512
# Postnet kernel size
postnet_kernel_size: int = 5
# Number of postnet convolutions
postnet_n_convolutions: int = 5
"""Attention parameters"""
# Number of units in attention LSTM
attention_rnn_dim: int = 1024
# Dimension of attention hidden representation
attention_dim: int = 128
"""Attention location parameters"""
# Number of filters for location-sensitive attention
attention_location_n_filters: int = 32
# Kernel size for location-sensitive attention
attention_location_kernel_size: int = 31
