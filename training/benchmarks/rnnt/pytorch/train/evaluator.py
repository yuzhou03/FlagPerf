import torch
import time
from contextlib import contextmanager

from common import helpers
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                           process_evaluation_epoch)
from common.tb_dllogger import flush_log, init_log, log
from utils.mlperf import logging


class Evaluator:

    def __init__(self, config, val_dataloader):
        self.config = config
        self.val_dataloader = val_dataloader

    def evaluate(self, trainer):
        state = trainer.training_state
        epoch = state.epoch
        step = state.step
        val_loader = trainer.val_loader
        detokenize = trainer.tokenizer.detokenize
        val_feat_proc = trainer.val_feat_proc
        ema_model = trainer.ema_model
        loss_fn = trainer.criterion
        greedy_decoder = trainer.greedy_decoder

        _evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
                  ema_model, loss_fn, greedy_decoder, False)


@torch.no_grad()
def _evaluate(epoch, step, val_loader, val_feat_proc, detokenize, ema_model,
              loss_fn, greedy_decoder, use_amp):

    ema_model.eval()

    start_time = time.time()
    agg = {'losses': [], 'preds': [], 'txts': [], 'idx': []}
    logging.log_start(logging.constants.EVAL_START,
                      metadata=dict(epoch_num=epoch))
    for i, batch in enumerate(val_loader):
        print(
            f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}',
            end='\r')

        audio, audio_lens, txt, txt_lens = batch

        feats, feat_lens = val_feat_proc([audio, audio_lens])

        log_probs, log_prob_lens = ema_model(feats, feat_lens, txt, txt_lens)
        loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                       log_prob_lens, txt, txt_lens)

        pred = greedy_decoder.decode(ema_model, feats, feat_lens)

        agg['losses'] += helpers.gather_losses([loss.cpu()])
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()],
                                                  [txt_lens.cpu()], detokenize)

    wer, loss = process_evaluation_epoch(agg)

    logging.log_event(logging.constants.EVAL_ACCURACY,
                      value=wer,
                      metadata=dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP,
                    metadata=dict(epoch_num=epoch))

    log((epoch, ), step, 'dev_ema', {
        'loss': loss,
        'wer': 100.0 * wer,
        'took': time.time() - start_time
    })
    ema_model.train()
    return wer
