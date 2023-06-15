from .models import GCN


def create_model(args, features, labels):

    print(f"create_model=== features shape:{features.shape} nclass:{labels.max().item() + 1}")
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    return model
