import torch.nn.functional as F
from utils.utils import accuracy


class Evaluator:

    def __init__(self, args, adj, features, labels, idx_test):
        self.args = args
        self.adj = adj
        self.idx_test = idx_test
        self.features = features
        self.labels = labels

    def evaluate(self, trainer):
        features = self.features
        labels = self.labels
        idx_test = self.idx_test

        if self.args.cuda:
            features = features.cuda()
            labels = labels.cuda()
            idx_test = idx_test.cuda()

        model = trainer.model
        model.eval()
        output = model(features, self.adj)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), loss_test.item()
