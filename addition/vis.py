from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback
from lightning.pytorch.callbacks import Callback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime
import warnings
import hydra
warnings.simplefilter(action="ignore", category=FutureWarning)

@hydra.main(
    config_path="config",
    config_name="search",
    version_base=None,
)
def main(cfg: DictConfig):
    out_dir = Path(cfg.convert_hf.out_path)

    state_dict = torch.load(out_dir / "model.pth")
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )

    