import torch
import torch.distributed as dist

from driver import dist_pytorch


class Evaluator:

    def __init__(self, args, dataloader, adj):
        self.args = args
        self.dataloader = dataloader
        self.adj = adj

        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_size = 0

    def __update(self, loss, acc, n):
        self.total_loss += loss * n
        self.total_acc += acc * n
        self.total_size += n

    def evaluate(self, trainer, is_testing: bool = False):
        self.total_loss, self.total_acc = 0.0, 0.0
        self.total_size = 0

        print(f"evaluating. is_testing:{is_testing}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch = trainer.process_batch(batch, self.args.device)
                _, loss, acc = trainer.inference(batch, batch_idx, self.adj,
                                                 is_testing)
                self.__update(loss.item(), acc.item(), batch[0].shape[0])

                print(
                    f"evaluate_in_batch, loss = {loss.item()}, acc = {acc.item()}, n = {batch[0].shape[0]}"
                )

        if dist_pytorch.is_dist_avail_and_initialized():
            total = torch.tensor(
                [self.total_loss, self.total_acc, self.total_size],
                dtype=torch.float32,
                device=self.args.device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            self.total_loss, self.total_acc, self.total_size = total.tolist()

        loss = self.total_loss / self.total_size
        acc = self.total_acc / self.total_size
        print(
            f"evaluate end. total_loss:{self.total_loss} total_acc:{self.total_acc} total_size:{self.total_size} loss:{loss} acc:{acc}"
        )

        return loss, acc
