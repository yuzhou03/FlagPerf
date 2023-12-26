# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from dataclasses import dataclass


@dataclass
class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    global_steps = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    wer: float = 0.0  # word error rate

    epoch: int = 0
    end_training: bool = False
    converged: bool = False

    accumulated_batches: int = 0

    init_time = 0
    raw_train_time = 0
    no_eval_time = 0
    pure_compute_time = 0

    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.end_training = True
        self.converged = True
