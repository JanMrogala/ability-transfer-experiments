"""
Tokenize preprocessed data using the specified tokenizer.
Saves tokenized data in an output directory based on data type and tokenizer.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch

from transformers import GPT2TokenizerFast
from datasets import Dataset, DatasetDict


def load_tokenizer(tokenizer_path):
    """
    Load a tokenizer from the specified path.
    
    Args:
        tokenizer_path: Path to the tokenizer
        
    Returns:
        The loaded tokenizer
    """
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded with: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}")
    return tokenizer


def load_data(file_path):
    """
    Load text data from a file, splitting by <|endoftext|> tokens.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of text entries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split by the endoftext token and filter out empty entries
        entries = [text.strip() for text in content.split('<|endoftext|>') if text.strip()]
        # Return raw entries without adding tokens - tokenizer will handle this
        return entries


def tokenize_data(texts, tokenizer, config, num_workers=32):
    """
    Tokenize data using Hugging Face Datasets for parallelization.
    
    Args:
        texts: List of text entries
        tokenizer: The tokenizer to use
        config: Tokenization configuration
        num_workers: Number of parallel workers
        
    Returns:
        Tokenized dataset
    """
    # Create a Dataset from the texts
    dataset = Dataset.from_dict({"text": texts})
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=config.tokenized_data.truncation,
            padding="max_length",
            max_length=config.tokenized_data.context_length,
            return_tensors="pt"
        )
    
    # Apply tokenization with parallel processing
    print(f"Tokenizing {len(texts)} texts using {num_workers} workers")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="Tokenizing texts"
    )
    
    return tokenized_dataset


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Tokenize data using the specified tokenizer.
    """
    # Set number of workers for tokenization
    num_workers = cfg.tokenizer.get('num_workers', 4)  # Default to 4 workers if not specified
    
    # Define a base output directory
    base_output_dir = Path("data/tokenized")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process each language with its own tokenizer
    for lang in cfg.data.languages:
        print(f"\nProcessing language: {lang}")
        
        # Load the language-specific tokenizer
        print(f"Loading tokenizer for language: {lang}")
        tokenizer_path = Path(cfg.tokenizer.output_dir) / f"tokenizer"
        
        # Print more debugging info
        print(f"Looking for tokenizer at: {tokenizer_path}")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer for {lang} not found at {tokenizer_path}. Please create it first.")
        
        tokenizer = load_tokenizer(tokenizer_path)
        
        # Create dataset dict for this language
        dataset_dict = DatasetDict()
        
        # Directly construct the file paths
        train_file = Path(cfg.data.datapath) / lang / "train.txt"
        val_file = Path(cfg.data.datapath) / lang / "val.txt"
        
        print(f"Looking for train file at: {train_file}")
        print(f"Looking for val file at: {val_file}")
        
        # Process training data
        if train_file.exists():
            print(f"Tokenizing training data from {train_file}")
            train_texts = load_data(train_file)
            print(f"Loaded {len(train_texts)} training examples")
            
            train_dataset = tokenize_data(train_texts, tokenizer, cfg, num_workers)
            dataset_dict["train"] = train_dataset
        else:
            print(f"Warning: Training file not found at {train_file}")
        
        # Process validation data
        if val_file.exists():
            print(f"Tokenizing validation data from {val_file}")
            val_texts = load_data(val_file)
            print(f"Loaded {len(val_texts)} validation examples")
            
            val_dataset = tokenize_data(val_texts, tokenizer, cfg, num_workers)
            dataset_dict["validation"] = val_dataset
        else:
            print(f"Warning: Validation file not found at {val_file}")
        
        # Define a specific output directory for this language
        # Format: data/tokenized/[language]
        output_dir = base_output_dir / lang
        os.makedirs(output_dir, exist_ok=True)
        
        # Save this language's dataset
        dataset_dict.save_to_disk(str(output_dir))
        print(f"Saved tokenized data for {lang} to {output_dir}")
        
        print(f"Completed processing for language: {lang}")

    print("Tokenization process completed successfully!")


if __name__ == "__main__":
    main()