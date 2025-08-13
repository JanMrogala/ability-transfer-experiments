"""
Create and train a BPE tokenizer for language model training.
The tokenizer is saved as tokenizer_{lang}.json.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import GPT2TokenizerFast


def create_tokenizer(data_files, vocab_size, special_tokens, bos_token, eos_token, pad_token, output_file=None):
    """
    Create and train a BPE tokenizer on the given data files.
    
    Args:
        data_files: List of files to train the tokenizer on
        vocab_size: Size of the vocabulary
        special_tokens: List of special tokens to add
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        pad_token: Padding token
        output_file: Path to save the tokenizer
        
    Returns:
        The trained tokenizer
    """
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Create a BPE trainer with the specified vocabulary size
    # Convert special_tokens from ListConfig to regular Python list
    special_tokens_list = list(special_tokens)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens_list,
        show_progress=True
    )
    
    # Read text data from files
    print(f"Training tokenizer on files: {data_files}")
    
    # Train the tokenizer on the data files
    tokenizer.train(files=data_files, trainer=trainer)
    
    # Configure post-processing and decoding
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.enable_padding(pad_token=pad_token)
    
    # Wrap the tokenizer to make it compatible with transformers
    wrapped_tokenizer = GPT2TokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=eos_token,
        bos_token=bos_token,
        pad_token=pad_token
    )
    
    # Save the tokenizer if output file is specified
    if output_file:
        print(f"Saving tokenizer to {output_file}")
        wrapped_tokenizer.save_pretrained(output_file)
    
    return wrapped_tokenizer


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Create tokenizers for all configured languages.
    """
    # Print config for debugging
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    lang = "en"
    print(f"Creating tokenizer for language: {lang} and cs")
    
    # Correctly resolve the path for this language
    train_file_path = f"{cfg.data.datapath}/{lang}/train.txt"
    val_file_path = f"{cfg.data.datapath}/{lang}/val.txt"
    
    train_file = Path(train_file_path)
    val_file = Path(val_file_path)
    
    # Collect all existing files to train on
    training_files = []

    training_files.append(str(Path(f"{cfg.data.datapath}/cs/train.txt")))
    training_files.append(str(Path(f"{cfg.data.datapath}/cs/val.txt")))

    if train_file.exists():
        print(f"Found train file: {train_file}")
        training_files.append(str(train_file))
    else:
        print(f"Warning: Train file not found at {train_file}")
    
    if val_file.exists():
        print(f"Found validation file: {val_file}")
        training_files.append(str(val_file))
    else:
        print(f"Warning: Validation file not found at {val_file}")
    
    
    # Ensure output directory exists
    os.makedirs(cfg.tokenizer.output_dir, exist_ok=True)
    
    # Output file path
    output_file = Path(cfg.tokenizer.output_dir) / f"tokenizer"
    
    # Create and train the tokenizer
    create_tokenizer(
        data_files=training_files,
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=cfg.tokenizer.special_tokens,
        bos_token=cfg.tokenizer.bos_token,
        eos_token=cfg.tokenizer.eos_token,
        pad_token=cfg.tokenizer.pad_token,
        output_file=str(output_file)
    )
    
    print(f"Tokenizer creation completed for {lang}. Saved to {output_file}")


if __name__ == "__main__":
    main()