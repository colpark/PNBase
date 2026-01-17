from typing import Any, Dict

import torch
import torch.nn as nn
import threading
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.utils.copy import swap_tensors

class SimpleEMA(Callback):
    def __init__(self,
                 decay: float = 0.9999,
                 every_n_steps: int = 1,
                 ):
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._stream = None  # Lazily initialized on first use
        self.previous_step = 0
        self.net_params = None
        self.ema_params = None

    def setup_models(self, net: nn.Module, ema_net: nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def ema_step(self):
        # Safety check: ensure setup_models was called
        if self.net_params is None or self.ema_params is None:
            raise RuntimeError("EMA callback: setup_models() must be called before ema_step()")

        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            torch._foreach_mul_(ema_model_tuple, decay)
            torch._foreach_add_(
                ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
            )

        # Check if we're on CUDA and lazily initialize stream
        use_cuda = self.ema_params[0].is_cuda
        if use_cuda:
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                ema_update(self.ema_params, self.net_params, self.decay)
            # Synchronize to ensure EMA update completes before any checkpoint save
            self._stream.synchronize()
        else:
            # CPU path - no stream needed
            ema_update(self.ema_params, self.net_params, self.decay)

        assert self.ema_params[0].dtype == torch.float32

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step == self.previous_step:
            return
        self.previous_step = trainer.global_step
        if trainer.global_step % self.every_n_steps == 0:
            self.ema_step()


    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]

