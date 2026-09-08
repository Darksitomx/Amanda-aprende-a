"""Trainer de SFT (§34, §37).

Loop de entrenamiento con CPU/GPU automático, mixed precision (fp16), gradient
accumulation, checkpoints, reanudación y registro de métricas.
"""
from __future__ import annotations

import json
import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressed_llm.config import Config
from compressed_llm.utils import set_seed

logger = logging.getLogger("compressed_llm.trainer")


class Trainer:
    """Configura y ejecuta el loop de entrenamiento SFT."""

    def __init__(self, model, cfg: Config, train_loader: DataLoader,
                 valid_loader: DataLoader = None, resume_from: str = None,
                 device: str = None):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        t = cfg.training

        set_seed(t.seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.use_amp = (t.mixed_precision == "fp16" and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
        )
        self.global_step = 0

        if resume_from and os.path.exists(resume_from):
            self._resume(resume_from)

    # ------------------------------------------------------------------ #
    def _resume(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            except ValueError:
                logger.warning("Optimizer no reanudado")
        self.global_step = payload.get("step", 0)
        logger.info("Reanudado en step=%d", self.global_step)

    def _valid_loss(self) -> float:
        if self.valid_loader is None:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                total += out["loss"].item() * batch["context_ids"].size(0)
                n += batch["context_ids"].size(0)
        self.model.train()
        return total / max(1, n)

    # ------------------------------------------------------------------ #
    def train(self, max_steps: int = None, log_steps: int = None,
              save_steps: int = None, output_dir: str = None) -> dict:
        t = self.cfg.training
        max_steps = max_steps or t.max_steps
        log_steps = log_steps or t.log_steps
        save_steps = save_steps or t.save_steps
        output_dir = output_dir or t.output_dir
        os.makedirs(output_dir, exist_ok=True)

        warmup = t.warmup_steps
        lr = t.learning_rate
        accum = t.gradient_accumulation
        metrics = []
        iterator = iter(self.train_loader)
        pbar = tqdm(total=max_steps, initial=self.global_step)
        step_lr = lr

        while self.global_step < max_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=self.use_amp,
                                        dtype=torch.float16):
                    out = self.model(**batch)
                    loss = out["loss"] / accum
                self.scaler.scale(loss).backward()
                accum_loss += out["loss"].item()

            step = self.global_step + 1
            scale = min(1.0, step / max(1, warmup))
            step_lr = lr * scale
            for pg in self.optimizer.param_groups:
                pg["lr"] = step_lr

            if t.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step = step

            if step % log_steps == 0 or step == max_steps:
                vloss = self._valid_loss()
                rec = {
                    "step": step,
                    "train_loss": accum_loss / accum,
                    "valid_loss": vloss,
                    "perplexity": math.exp(min(accum_loss / accum, 20.0)),
                    "lr": step_lr,
                }
                metrics.append(rec)
                logger.info("step=%d loss=%.4f ppl=%.2f lr=%.2e",
                            step, rec["train_loss"], rec["perplexity"], rec["lr"])

            if step % save_steps == 0 or step == max_steps:
                self._save_checkpoint(os.path.join(output_dir, f"step_{step}.pt"), step)
            pbar.update(1)

        pbar.close()
        with open(os.path.join(output_dir, "metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return {"metrics": metrics, "final_step": self.global_step}

    def _save_checkpoint(self, path: str, step: int) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": step,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        logger.info("Checkpoint guardado en %s", path)