import functions as ag
import os
import json
import random
import argparse

def create_parser():
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(description='Generate datasets with specific format(s)')
    parser.add_argument('--format', type=str, default='all',
                        help='Format to generate. Options: all, original, hex, binary, roman, letter')
    return parser

# Create the main data directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/lengths_analysis", exist_ok=True)

# Define all available format configurations
ALL_FORMATS = [
    {
        "name": "original",
        "generator": ag.generate_original_format,
        "dir": "data/lengths_analysis/original"
    },
    {
        "name": "hex",
        "generator": ag.generate_hex_format,
        "dir": "data/lengths_analysis/hex"
    },
    {
        "name": "binary",
        "generator": ag.generate_binary_format,
        "dir": "data/lengths_analysis/binary"
    },
    {
        "name": "roman",
        "generator": ag.generate_roman_format,
        "dir": "data/lengths_analysis/roman"
    },
    {
        "name": "letter",
        "generator": ag.generate_letter_format,
        "dir": "data/lengths_analysis/letter"
    }
]

def get_selected_formats(format_name):
    """Filter formats based on user selection."""
    if format_name == 'all':
        return ALL_FORMATS
    else:
        selected = [f for f in ALL_FORMATS if f["name"] == format_name]
        if not selected:
            raise ValueError(f"Unknown format: {format_name}. Available formats: " + 
                             ", ".join([f["name"] for f in ALL_FORMATS]))
        return selected

# Define training set sizes (unique examples before oversampling)
train_sizes = [512, 1024, 2048, 4096, 8192, 12800, 16384, 25600]

# Target size after oversampling (all datasets will have EXACTLY this many examples)
target_train_size = 25600

# Define a common test set size
test_size = 1024

# Define token length range for the results
min_tokens = 3
max_tokens = 10

# Set a base random seed (will be varied per format)
base_random_seed = 42

# IMPORTANT: Maintain our own global registry of examples to prevent overlap
# This will store ALL examples as (input, output) tuples
all_test_examples = set()
all_train_examples = set()

def example_to_tuple(example):
    """Convert an example dict to a hashable tuple."""
    return (example["input"], example["output"])

def create_exact_oversampled_dataset(unique_examples, target_size, output_file):
    """
    Create a dataset with exactly target_size examples by oversampling from unique_examples.
    
    Args:
        unique_examples: List of example dictionaries
        target_size: Exact number of examples wanted in the final dataset
        output_file: Path to save the oversampled dataset
    """
    # Calculate how many full repetitions we need
    num_unique = len(unique_examples)
    full_repetitions = target_size // num_unique
    
    # Calculate how many additional examples we need
    additional_examples = target_size % num_unique
    
    # Create the oversampled dataset
    oversampled_dataset = []
    
    # Add full repetitions
    for _ in range(full_repetitions):
        oversampled_dataset.extend(unique_examples)
    
    # Add additional examples to reach exactly target_size
    if additional_examples > 0:
        # Use a copy to avoid modifying the original when shuffling
        extra_examples = unique_examples.copy()
        random.shuffle(extra_examples)
        oversampled_dataset.extend(extra_examples[:additional_examples])
    
    # Double-check we have exactly the right number
    assert len(oversampled_dataset) == target_size, f"Expected {target_size} examples, got {len(oversampled_dataset)}"
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(oversampled_dataset, f, indent=2)
    
    return len(oversampled_dataset)

def generate_unique_examples(generator, num_examples, for_test=False, min_tokens=3, max_tokens=10, random_seed=None):
    """
    Generate unique examples that don't overlap with existing examples.
    
    Args:
        generator: The generator function to use (e.g., ag.generate_original_format)
        num_examples: Number of examples to generate
        for_test: Whether these examples are for the test set or train set
        min_tokens, max_tokens: Token length parameters
        random_seed: Random seed for reproducibility
    
    Returns:
        A list of unique example dictionaries
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    examples = []
    attempts = 0
    max_attempts = num_examples * 100  # Limit attempts to avoid infinite loops
    
    print(f"Generating {num_examples} unique {'test' if for_test else 'training'} examples")
    
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        
        # Generate a single example
        if generator.__name__ in ['generate_original_format', 'generate_roman_format', 
                                'generate_binary_format', 'generate_hex_format',
                                'generate_letter_format']:
            example = generator(1, min_tokens=min_tokens, max_tokens=max_tokens)[0]
        else:
            example = generator(1)  # For generators that don't support token range
            
        example_tuple = example_to_tuple(example)
        
        # For test examples, check only against existing test examples
        if for_test:
            if example_tuple not in all_test_examples:
                examples.append(example)
                all_test_examples.add(example_tuple)
        # For train examples, check against BOTH test and train to ensure no overlap
        else:
            if (example_tuple not in all_train_examples and 
                example_tuple not in all_test_examples):
                examples.append(example)
                all_train_examples.add(example_tuple)
        
        # Print progress occasionally
        if attempts % 1000 == 0:
            print(f"  Progress: {len(examples)}/{num_examples} after {attempts} attempts")
    
    if len(examples) < num_examples:
        print(f"WARNING: Could only generate {len(examples)} unique examples, requested {num_examples}")
    
    return examples

def verify_no_overlap():
    """Check if there is any overlap between test and train examples."""
    overlap = all_test_examples.intersection(all_train_examples)
    if overlap:
        print(f"ERROR: Found {len(overlap)} examples that appear in both train and test sets!")
        print("First few overlapping examples:")
        for i, ex in enumerate(list(overlap)[:3]):
            print(f"  Overlap {i+1}: {ex}")
        return False
    else:
        print("✓ VERIFIED: No overlap between train and test sets")
        return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Get the selected formats
    formats = get_selected_formats(args.format)
    
    print(f"\nSelected format(s): {', '.join([f['name'] for f in formats])}")
    
    print("\n===== PHASE 1: GENERATING TEST SETS =====\n")
    
    # FIRST PHASE: Generate all test sets to prevent overlap
    for format_idx, format_config in enumerate(formats):
        print(f"\n{'='*50}")
        print(f"Generating test set for {format_config['name']} format")
        print(f"{'='*50}")
        
        # Create the format directory
        os.makedirs(format_config['dir'], exist_ok=True)
        
        # Use a different seed for each format
        format_seed = base_random_seed + format_idx * 100
        
        # Path for test file
        test_output_file = os.path.join(format_config['dir'], "test.json")
        
        # Check if test set already exists
        if os.path.exists(test_output_file):
            print(f"Test set for {format_config['name']} already exists, loading into cache")
            
            # Load existing test set into our global registry
            with open(test_output_file, 'r') as f:
                existing_test = json.load(f)
                
            for example in existing_test:
                all_test_examples.add(example_to_tuple(example))
                
            print(f"Loaded {len(existing_test)} test examples into cache")
                
            # Remove any unwanted train.json in format directory
            unwanted_train_file = os.path.join(format_config['dir'], "train.json")
            if os.path.exists(unwanted_train_file):
                os.remove(unwanted_train_file)
                print(f"Removed existing unwanted {unwanted_train_file}")
        else:
            # Generate a fresh test set
            test_examples = generate_unique_examples(
                generator=format_config['generator'],
                num_examples=test_size,
                for_test=True,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                random_seed=format_seed
            )
            
            # Write to file
            with open(test_output_file, 'w') as f:
                json.dump(test_examples, f, indent=2)
                
            print(f"Created test set with {len(test_examples)} examples")
    
    # Verify after generating all test sets
    print("\nAfter test set generation:")
    print(f"Total unique test examples across selected format(s): {len(all_test_examples)}")
    
    print("\n===== PHASE 2: GENERATING TRAINING SETS =====\n")
    
    # SECOND PHASE: Generate all training sets
    for format_idx, format_config in enumerate(formats):
        print(f"\n{'='*50}")
        print(f"Generating training sets for {format_config['name']} format")
        print(f"{'='*50}")
        
        # Use a different seed for each format
        format_seed = base_random_seed + 500 + format_idx * 100
        
        # Loop through each training size
        for size_idx, train_size in enumerate(train_sizes):
            # Use a different seed for each size
            size_seed = format_seed + size_idx * 10
            
            # Create a size-specific subdirectory
            train_dir = os.path.join(format_config['dir'], str(train_size))
            os.makedirs(train_dir, exist_ok=True)
            
            # Path to the train.json file
            train_output_file = os.path.join(train_dir, "train.json")
            
            # Path to a temporary file for unique examples
            unique_examples_file = os.path.join(train_dir, "unique_examples_temp.json")
            
            # Check if this train set already exists
            if os.path.exists(train_output_file):
                print(f"Train set of size {train_size} for {format_config['name']} already exists")
                
                # Load existing train set into our global registry to prevent future duplicates
                with open(train_output_file, 'r') as f:
                    existing_train = json.load(f)
                
                # Extract only the unique examples (before oversampling)
                unique_examples = []
                seen = set()
                for example in existing_train:
                    example_tuple = example_to_tuple(example)
                    if example_tuple not in seen:
                        unique_examples.append(example)
                        seen.add(example_tuple)
                        all_train_examples.add(example_tuple)
                
                print(f"Loaded {len(unique_examples)} unique training examples into cache")
                continue
            
            # Generate unique training examples (guaranteed no overlap with test set)
            unique_train_examples = generate_unique_examples(
                generator=format_config['generator'],
                num_examples=train_size,
                for_test=False,  # This is for training
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                random_seed=size_seed
            )
            
            # Save the unique examples to a temporary file
            with open(unique_examples_file, 'w') as f:
                json.dump(unique_train_examples, f, indent=2)
            
            # Verify NO overlap with test set
            overlap = set(example_to_tuple(ex) for ex in unique_train_examples).intersection(all_test_examples)
            if overlap:
                print(f"ERROR: {len(overlap)} examples overlap with test set! This should not happen!")
            else:
                print(f"✓ Verified: No overlap between these {len(unique_train_examples)} train examples and test set")
            
            # Special case: if train_size is already target_train_size, no oversampling needed
            if train_size == target_train_size:
                # Just rename the file to train.json
                os.rename(unique_examples_file, train_output_file)
                print(f"No oversampling needed for {train_size} examples, already at target size")
            else:
                # Create the exactly oversampled dataset
                print(f"Oversampling {len(unique_train_examples)} unique examples to exactly {target_train_size} total examples")
                total_examples = create_exact_oversampled_dataset(
                    unique_train_examples, 
                    target_train_size, 
                    train_output_file
                )
                
                # Remove the temporary file of unique examples
                os.remove(unique_examples_file)
                
                print(f"Created train set with exactly {total_examples} examples")
    
    # Final verification
    verify_no_overlap()
    
    print("\nDataset generation complete!")
    print(f"Created datasets for formats: {', '.join([f['name'] for f in formats])}")
    print(f"Training set unique sizes: {', '.join([str(s) for s in train_sizes])}")
    print(f"All training sets oversampled to exactly {target_train_size} total examples")
    print(f"Test set size: {test_size}")
    print(f"Token length range: {min_tokens}-{max_tokens}")
    print(f"Total unique test examples: {len(all_test_examples)}")
    print(f"Total unique train examples: {len(all_train_examples)}")