from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from utils.utils import load_data



def gpu_load_data(args):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        path=args.data_dir, dataset=args.dataset)
    return adj, features, labels, idx_train, idx_val, idx_test


def build_train_dataset(args):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = gpu_load_data(args)

    train_dataset = TensorDataset(features[idx_train], labels[idx_train])
    return train_dataset


def build_train_dataloader(args, train_dataset):
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, seed=args.seed)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        pin_memory=False,
    )
    return train_dataloader


def build_eval_dataset(args):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = gpu_load_data(args)
    dataset = TensorDataset(features[idx_val], labels[idx_val])
    return dataset


def build_eval_dataloader(args, val_dataset):
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = DataLoader(val_dataset,
                                num_workers=args.num_workers,
                                shuffle=False,
                                sampler=val_sampler,
                                batch_size=args.eval_batch_size,
                                pin_memory=False)
    return val_dataloader
