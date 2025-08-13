"""
Mix tokenized datasets from two languages with specified proportions.
Creates a new mixed dataset containing all data from main language plus 
a specified percentage of data from the secondary language.
"""

import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
import random
from datasets import Dataset, DatasetDict, concatenate_datasets


def load_tokenized_dataset(dataset_path):
    """
    Load a tokenized dataset from disk.
    
    Args:
        dataset_path: Path to the tokenized dataset directory
        
    Returns:
        DatasetDict containing the loaded dataset
    """
    print(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    dataset_dict = DatasetDict.load_from_disk(str(dataset_path))
    return dataset_dict


def sample_dataset(dataset, proportion):
    """
    Sample a specified proportion of examples from a dataset.
    
    Args:
        dataset: The dataset to sample from
        proportion: Percentage (0-100) of data to sample
        
    Returns:
        Sampled dataset
    """
    total_examples = len(dataset)
    sample_count = int(total_examples * proportion / 100)
    
    # Create shuffled indices for random sampling
    indices = list(range(total_examples))
    random.shuffle(indices)
    selected_indices = indices[:sample_count]
    
    # Select the subset
    sampled_dataset = dataset.select(selected_indices)
    
    print(f"Sampled {sample_count} examples from {total_examples} total ({proportion}%)")
    return sampled_dataset


def create_mixed_dataset(main_dataset_dict, secondary_dataset_dict, proportion, secondary_lang):
    """
    Create a mixed dataset combining main language data with a proportion of secondary language data.
    
    Args:
        main_dataset_dict: Main language dataset dictionary
        secondary_dataset_dict: Secondary language dataset dictionary  
        proportion: Percentage of secondary language data to include
        secondary_lang: Name of secondary language for logging
        
    Returns:
        Mixed DatasetDict
    """
    mixed_dataset_dict = DatasetDict()
    
    # Process each split (train, validation)
    for split in main_dataset_dict.keys():
        print(f"\nProcessing {split} split:")
        
        # Get all main language data
        main_data = main_dataset_dict[split]
        main_count = len(main_data)
        print(f"Main language examples: {main_count}")
        
        # Get secondary language data if split exists
        if split in secondary_dataset_dict:
            secondary_data = secondary_dataset_dict[split]
            secondary_total = len(secondary_data)
            print(f"Secondary language ({secondary_lang}) total examples: {secondary_total}")
            
            # Sample proportion of secondary data
            if proportion > 0:
                secondary_data_subset = sample_dataset(secondary_data, proportion)
                secondary_count = len(secondary_data_subset)
                
                # Combine datasets
                combined_data = concatenate_datasets([main_data, secondary_data_subset])
                print(f"Combined dataset size: {len(combined_data)} (main: {main_count} + secondary: {secondary_count})")
            else:
                combined_data = main_data
                print(f"No secondary data added (proportion = 0)")
                
        else:
            print(f"Warning: {split} split not found in secondary language dataset")
            combined_data = main_data
        
        # Shuffle the combined dataset for better mixing
        combined_data = combined_data.shuffle(seed=42)
        mixed_dataset_dict[split] = combined_data
    
    return mixed_dataset_dict


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Mix tokenized datasets from two languages according to configuration.
    """
    # Get mix configuration
    main_lang = cfg.mix.main_lang
    proportion = cfg.mix.proportion
    
    print(f"Creating mixed dataset with main language: {main_lang}, proportion: {proportion}%")
    
    # Determine secondary language
    available_langs = cfg.data.languages
    secondary_langs = [lang for lang in available_langs if lang != main_lang]
    
    if not secondary_langs:
        raise ValueError(f"No secondary language found. Available languages: {available_langs}")
    
    secondary_lang = secondary_langs[0]  # Take the first available secondary language
    print(f"Secondary language: {secondary_lang}")
    
    # Define paths
    base_tokenized_dir = Path("data/tokenized")
    main_lang_dir = base_tokenized_dir / main_lang
    secondary_lang_dir = base_tokenized_dir / secondary_lang
    
    # Validate paths exist
    if not main_lang_dir.exists():
        raise FileNotFoundError(f"Main language tokenized data not found at {main_lang_dir}")
    if not secondary_lang_dir.exists():
        raise FileNotFoundError(f"Secondary language tokenized data not found at {secondary_lang_dir}")
    
    # Load datasets
    main_dataset_dict = load_tokenized_dataset(main_lang_dir)
    secondary_dataset_dict = load_tokenized_dataset(secondary_lang_dir)
    
    # Create mixed dataset
    mixed_dataset_dict = create_mixed_dataset(
        main_dataset_dict, 
        secondary_dataset_dict, 
        proportion, 
        secondary_lang
    )
    
    # Create output directory and save
    output_dir = base_tokenized_dir / f"mix_{proportion}_{secondary_lang}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving mixed dataset to {output_dir}")
    # Save with optimized settings for speed
    mixed_dataset_dict.save_to_disk(
        str(output_dir),
        max_shard_size="500MB",  # Smaller shards for faster saving
        num_proc=16  # Use multiple processes for saving
    )
    
    # Print final statistics
    print("\nFinal mixed dataset statistics:")
    for split, dataset in mixed_dataset_dict.items():
        print(f"  {split}: {len(dataset)} examples")
    
    print(f"\nMixed dataset creation completed successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()