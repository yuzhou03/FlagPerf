vendor: str = "nvidia"
data_dir: str = "/mnt/data/maskrcnn/train/"
<<<<<<< HEAD
train_batch_size = 16
eval_batch_size = 16
=======
train_batch_size = 8
eval_batch_size = 8
>>>>>>> 913d4a1d978ef456b6d5a68ab094e0fe2bdac454

dist_backend = "nccl"
weight_decay = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-08
gradient_accumulation_steps = 1
warmup = 0.1
<<<<<<< HEAD
lr = 0.01
=======
lr = 0.02
>>>>>>> 913d4a1d978ef456b6d5a68ab094e0fe2bdac454
log_freq = 1
seed = 10483
max_samples_termination = 5553080
training_event = None

# resume: str = "output/checkpoint/model_13.pth"
# start_epoch: int = 13