import torch.nn.functional as F
from utils.utils import accuracy


class Evaluator:

    def __init__(self, args, features, labels, adj, idx_test):
        self.args = args
        self.features = features
        self.labels = labels
        self.adj = adj
        self.idx_test = idx_test

    def evaluate(self, trainer):
        labels = self.labels
        idx_test = self.idx_test

        model = trainer.model
        model.eval()
        output = model(self.features, self.adj)

        if self.args.cuda:
            labels = labels.cuda()
            idx_test = idx_test.cuda()

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), loss_test.item()
