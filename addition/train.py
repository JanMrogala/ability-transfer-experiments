# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from config import hf_config
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.evaluator import Evaluator
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import json
import os
import wandb
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from typing import Optional

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting training...")


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, train_batches, delimiter_token_id, trainer_ckpt_path=None):
        super().__init__()

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.trainer_ckpt_path = trainer_ckpt_path
        self.train_batches = train_batches
        self.delimiter_token_id = delimiter_token_id
        _, self.hf_conf = hf_config.get_configs(cfg)
        
        # Store dataset names for validation reporting
        self.dataset_names = []
        if hasattr(cfg.data, 'test_files'):
            for test_file in cfg.data.test_files:
                base_name = os.path.splitext(os.path.basename(test_file))[0]
                self.dataset_names.append(base_name)
        
        # For tracking combined validation metrics
        self.val_acc_sum = 0.0
        self.val_count = 0

    def setup(self, stage):
        self.hf_conf["bos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[BOS]")
        self.hf_conf["eos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[EOS]")
        self.hf_conf["vocab_size"] = len(self.preprocessor.tokenizer.get_vocab())

        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        # Find positions with delimiter tokens
        delimiter_positions = (input_ids == self.delimiter_token_id)
        
        # Create a shifted version where positions after delimiter are marked
        # This will include the delimiter itself as True
        first_search_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        first_search_pos[:, 1:] = delimiter_positions.cumsum(dim=1)[:, :-1].bool()
        
        # Create the mask - True for delimiter and positions before it
        mask = ~first_search_pos.cumsum(dim=1).bool()
        
        # Apply the mask to targets, setting masked positions to -100
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets_no_mask, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets_no_mask)
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        # Reset tracking variables at the start of each validation epoch
        self.val_acc_sum = 0.0
        self.val_count = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets)
        out, loss = self(idx, targets)
        
        # Compute accuracy only on non-masked tokens
        predictions = torch.argmax(out, dim=-1)
        
        # Create mask for non-masked tokens in the shifted targets
        valid_mask = (targets[:, 1:] != -100)
        
        # Compare predictions with targets only on non-masked positions
        correct = (predictions[:, :-1] == targets[:, 1:]) * valid_mask
        
        # Count total correct predictions and total valid tokens
        total_correct = correct.sum()
        total_tokens = valid_mask.sum()
        
        # Calculate accuracy (avoid division by zero)
        accuracy = total_correct.float() / total_tokens if total_tokens > 0 else torch.tensor(0.0)

        # Get the dataset name for this validation dataloader
        dataset_name = self.dataset_names[dataloader_idx] if dataloader_idx < len(self.dataset_names) else f"dataset_{dataloader_idx}"
        
        # Log metrics with dataset prefix for validation
        self.log(f"Validation/{dataset_name}/acc", accuracy, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"Validation/{dataset_name}/loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        
        # Track for combined accuracy (for checkpoint monitoring)
        self.val_acc_sum += accuracy.item()
        self.val_count += 1
        
        return {"val_loss": loss, "val_acc": accuracy}
    
    def on_validation_epoch_end(self):
        # Compute and log average validation accuracy (for checkpoint monitoring)
        if self.val_count > 0:
            avg_acc = self.val_acc_sum / self.val_count
            self.log("acc", avg_acc, prog_bar=True, sync_dist=True)
        
        # Save model checkpoint
        save_path = self.cfg.convert_hf.in_path
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)
        self.llm.model.to(self.device)
        
        # Run evaluation on each dataset
        self.run_evaluation()
    
    def run_evaluation(self):
        """Run evaluation on each dataset and log metrics."""
        # Evaluate on each dataset separately
        for dataset_idx, dataset_name in enumerate(self.dataset_names):
            # Get the appropriate test dataset
            test_key = f"test_{dataset_name}"
            if test_key in self.trainer.datamodule.dataset:
                test_data = self.trainer.datamodule.dataset[test_key]
                
                print(f"Running evaluation on dataset: {dataset_name}")
                
                # Evaluate the model on this dataset
                evaluator = Evaluator(
                    self.cfg,
                    test_data,
                    self.preprocessor.tokenizer,
                    self.cfg.data.split_str,
                    self.global_step,
                    self.llm.model,
                )
                
                # Get metrics dictionary from evaluator
                metrics = evaluator.evaluate()
                
                # Log all metrics to wandb/Lightning with dataset prefix
                for metric_name, value in metrics.items():
                    self.log(
                        f"Evaluation/{dataset_name}/{metric_name}",
                        value,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                
                if wandb.run is not None and hasattr(evaluator, 'predictions_after_delimiter'):
                    # Create example table for this dataset
                    examples_table = wandb.Table(columns=["Prompt", "Prediction", "Ground Truth", "Exact Match"])
                    
                    # Select a few random indices to log
                    num_examples = min(self.cfg.wandb.num_examples_reported, len(evaluator.prompts))
                    indices = np.random.choice(len(evaluator.prompts), num_examples, replace=False)
                    
                    for i in indices:
                        prompt = self.preprocessor.tokenizer.decode(evaluator.prompts[i], skip_special_tokens=True)
                        pred = evaluator.predictions_after_delimiter[i]
                        gt = evaluator.gts[i]
                        exact_match = pred == gt
                        
                        examples_table.add_data(prompt, pred, gt, exact_match)
                    
                    wandb.log({f"Examples/{dataset_name}": examples_table}, step=self.global_step)

    def configure_optimizers(self):
        if self.cfg.optim.lr_type == "linear":
            warmup_steps = self.cfg.optim.warmup_steps
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: (step + 1) / warmup_steps
            )
            return [optimizer], [scheduler]
        elif self.cfg.optim.lr_type == "linear-reg":
            warmup_steps = self.cfg.optim.warmup_steps
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: (step + 1) / warmup_steps
            )
            return [optimizer], [scheduler]
        
        elif self.cfg.optim.lr_type == "linear-reg-capped":
            warmup_steps = self.cfg.optim.warmup_steps
            max_lr = self.cfg.optim.max_lr
            base_lr = self.cfg.optim.lr  # Original learning rate (0.00002)
            
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=base_lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            
            # Define a lambda function that caps at max_lr
            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps * (max_lr / base_lr)
                else:
                    return max_lr / base_lr
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda
            )
            
            return [optimizer], [scheduler]
        else:
            # n_steps = self.cfg.optim.n_steps
            n_steps = self.cfg.model.epochs * self.train_batches
            warmup_steps = self.cfg.optim.warmup_steps

            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            scheduler = {
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=n_steps,
                ),
                "interval": "step",
            }
            return [optimizer], [scheduler]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)


@hydra.main(
    config_path="config",
    config_name="search",
    version_base=None,
)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers

    tokenizer = get_tokenizer(cfg)
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    conf.padded_vocab_size = len(tokenizer.get_vocab())
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)
    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)
    data.connect(max_seq_length=cfg.model.block_size)
    data.setup()

    train_size = len(data.train_dataloader())
    trace_start_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=True)[0]

    lit_model = LitLLM(model=model, cfg=cfg, train_batches=train_size, preprocessor=preprocessor,
                       delimiter_token_id=trace_start_token_id)

    logger = WandbLogger(
        project=cfg.wandb.proj_name, name=f"{cfg.model.name}", config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",  # This is the combined accuracy across validation sets
        dirpath=f"temp/checkpoints/{cfg.model.name}",
        filename="{epoch:02d}-{acc:.4f}",
        save_top_k=2,
        mode="max",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of params:", total_params)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()