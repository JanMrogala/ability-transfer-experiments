import json
import os
import random

DIGIT_TO_LETTER = {
    '0': 'A',
    '1': 'B',
    '2': 'C',
    '3': 'D',
    '4': 'E',
    '5': 'F',
    '6': 'G',
    '7': 'H',
    '8': 'I',
    '9': 'J'
}

def create_dirs(*dirs):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# ---- Utility functions ----

def convert_to_letters(num_str):
    """Convert a string of digits to letters."""
    return [DIGIT_TO_LETTER[digit] for digit in num_str]

def int_to_roman_tokens(num):
    """Convert an integer to a list of individual Roman numeral symbols (I, V, X, L, C, D, M)."""
    val_to_roman = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
        (1, "I")
    ]
    
    roman_str = ""
    i = 0
    
    while num > 0 and i < len(val_to_roman):
        symbol = val_to_roman[i][1]
        value = val_to_roman[i][0]
        
        if num >= value:
            roman_str += symbol
            num -= value
        else:
            i += 1
    
    # Split the Roman numeral string into individual characters
    roman_tokens = list(roman_str)
    
    return roman_tokens

def can_represent_in_n_roman_tokens(num, n=10):
    """Check if a number can be represented with n or fewer Roman numeral tokens."""
    return len(int_to_roman_tokens(num)) <= n

def has_n_to_m_roman_tokens(num, n=3, m=10):
    """Check if a number has between n and m Roman numeral tokens (inclusive)."""
    tokens = int_to_roman_tokens(num)
    return n <= len(tokens) <= m

def has_exactly_n_roman_tokens(num, n):
    """Check if a number can be represented with exactly n Roman numeral tokens."""
    tokens = int_to_roman_tokens(num)
    return len(tokens) == n

def int_to_binary_tokens(num):
    """Convert an integer to a list of binary tokens (individual digits)."""
    binary = bin(num)[2:]  # Remove '0b' prefix
    return list(binary)

def int_to_hex_tokens(num):
    """Convert an integer to a list of hexadecimal tokens (individual digits)."""
    hex_str = hex(num)[2:].upper()  # Remove '0x' prefix and convert to uppercase
    return list(hex_str)

def has_exactly_n_binary_digits(num, n):
    """Check if a number can be represented with exactly n binary digits."""
    binary = bin(num)[2:]  # Remove '0b' prefix
    return len(binary) == n

def has_exactly_n_hex_digits(num, n):
    """Check if a number can be represented with exactly n hexadecimal digits."""
    hex_str = hex(num)[2:]  # Remove '0x' prefix
    return len(hex_str) == n

def can_represent_in_n_binary_digits(num, n=10):
    """Check if a number can be represented with n or fewer binary digits."""
    binary = bin(num)[2:]  # Remove '0b' prefix
    return len(binary) <= n

def can_represent_in_n_hex_digits(num, n=10):
    """Check if a number can be represented with n or fewer hexadecimal digits."""
    hex_str = hex(num)[2:]  # Remove '0x' prefix
    return len(hex_str) <= n

# ---- Formatting functions ----

def format_original(a, b, c):
    """Format addition in original format: 1 2 3 4 5 + 1 2 3 4 5 = 5 4 3 2 1 (reversed)"""
    a_spaced = ' '.join(str(a))
    b_spaced = ' '.join(str(b))
    
    # Convert the number to a list of characters, reverse it, and join with spaces
    c_tokens = list(str(c))
    c_tokens_reversed = c_tokens[::-1]  # Reverse the tokens
    c_spaced = ' '.join(c_tokens_reversed)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_roman(a, b, c):
    """Format addition in Roman numeral format with spaces between tokens, with reversed output"""
    a_roman = int_to_roman_tokens(a)
    b_roman = int_to_roman_tokens(b)
    c_roman = int_to_roman_tokens(c)
    
    a_spaced = ' '.join(a_roman)
    b_spaced = ' '.join(b_roman)
    
    # Reverse the Roman numeral tokens
    c_roman_reversed = c_roman[::-1]
    c_spaced = ' '.join(c_roman_reversed)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_binary(a, b, c):
    """Format addition in binary format with spaces between tokens, with reversed output"""
    a_binary = int_to_binary_tokens(a)
    b_binary = int_to_binary_tokens(b)
    c_binary = int_to_binary_tokens(c)
    
    a_spaced = ' '.join(a_binary)
    b_spaced = ' '.join(b_binary)
    
    # Reverse the binary digits
    c_binary_reversed = c_binary[::-1]
    c_spaced = ' '.join(c_binary_reversed)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def remap_binary(input_text, output_text):
    input_text = input_text.replace("0", "O").replace("1", "I")
    output_text = output_text.replace("0", "O").replace("1", "I")

    return input_text, output_text

def format_hex(a, b, c):
    """Format addition in hexadecimal format with spaces between tokens, with reversed output"""
    a_hex = int_to_hex_tokens(a)
    b_hex = int_to_hex_tokens(b)
    c_hex = int_to_hex_tokens(c)
    
    a_spaced = ' '.join(a_hex)
    b_spaced = ' '.join(b_hex)
    
    # Reverse the hex digits
    c_hex_reversed = c_hex[::-1]
    c_spaced = ' '.join(c_hex_reversed)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_letter(a, b, c):
    """Format addition in letter format with exclamation mark: ! B C D E F , G H I J = F E D C B"""
    a_letters = convert_to_letters(str(a))
    b_letters = convert_to_letters(str(b))
    c_letters = convert_to_letters(str(c))
    
    a_spaced = ' '.join(a_letters)
    b_spaced = ' '.join(b_letters)
    
    # Reverse the result tokens
    c_letters_reversed = c_letters[::-1]
    c_spaced = ' '.join(c_letters_reversed)
    
    input_text = f"! {a_spaced} , {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_sorted_list(unsorted_list, sorted_list):
    """Format sorting task: unsorted list -> sorted list"""
    unsorted_str = ' '.join(str(num) for num in unsorted_list)
    sorted_str = ' '.join(str(num) for num in sorted_list)
    
    input_text = f"S {unsorted_str}"
    output_text = f"{sorted_str}"
    
    return input_text, output_text

# ---- Generator functions ----

def generate_sorting_format(num_examples):
    """Generate examples for sorting lists of random numbers."""
    examples = []
    
    while len(examples) < num_examples:
        list_length = random.randint(4, 10)
        unsorted_list = [random.randint(0, 9) for _ in range(list_length)]
        sorted_list = sorted(unsorted_list)
        
        input_text, output_text = format_sorted_list(unsorted_list, sorted_list)
        examples.append({
            "input": input_text,
            "output": output_text
        })
    
    return examples

def generate_letter_format(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in letter format."""
    examples = []
    
    while len(examples) < num_examples:
        # Choose a random target digit count for the sum
        target_digits = random.randint(min_tokens, max_tokens)
        
        # Determine maximum possible digits for addends based on target digits
        max_addend_digits = target_digits  # One addend can have as many digits as the target
        
        # Generate addends with appropriate digit counts
        a_digits = random.randint(1, max_addend_digits)
        b_digits = random.randint(1, max_addend_digits)
        
        # Generate the actual numbers with the specified number of digits
        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 1
        a_max = (10 ** a_digits) - 1
        a = random.randint(a_min, a_max)
        
        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 1
        b_max = (10 ** b_digits) - 1
        b = random.randint(b_min, b_max)
        
        # Calculate their sum
        c = a + b
        
        # Check if the sum has exactly the target number of digits
        if 10 ** (target_digits - 1) <= c < 10 ** target_digits:
            input_text, output_text = format_letter(a, b, c)
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples
    
def generate_original_format(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in original decimal format."""
    examples = []
    
    while len(examples) < num_examples:
        # Choose a random target digit count for the sum
        target_digits = random.randint(min_tokens, max_tokens)
        
        # Determine maximum possible digits for addends based on target digits
        # For a sum to have n digits, addends should have appropriate sizes
        max_addend_digits = target_digits  # One addend can have as many digits as the target
        
        # Generate addends with appropriate digit counts
        a_digits = random.randint(1, max_addend_digits)
        b_digits = random.randint(1, max_addend_digits)
        
        # Generate the actual numbers with the specified number of digits
        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 1
        a_max = (10 ** a_digits) - 1
        a = random.randint(a_min, a_max)
        
        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 1
        b_max = (10 ** b_digits) - 1
        b = random.randint(b_min, b_max)
        
        # Calculate their sum
        c = a + b
        
        # Check if the sum has exactly the target number of digits
        if 10 ** (target_digits - 1) <= c < 10 ** target_digits:
            input_text, output_text = format_original(a, b, c)
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples

def generate_roman_format(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in Roman numeral format."""
    examples = []
    
    possible_sums = []
    for i in range(1, 10000):
        tokens = int_to_roman_tokens(i)
        if min_tokens <= len(tokens) <= max_tokens:
            possible_sums.append(i)
    
    if not possible_sums:
        raise ValueError(f"Could not find any numbers representable with {min_tokens}-{max_tokens} Roman tokens")
    
    print(f"Found {len(possible_sums)} numbers representable with {min_tokens}-{max_tokens} Roman tokens")
    
    while len(examples) < num_examples:
        target_tokens = random.randint(min_tokens, max_tokens)
        target_sums = [num for num in possible_sums if len(int_to_roman_tokens(num)) == target_tokens]
        
        if not target_sums:
            continue
            
        c = random.choice(target_sums)
        
        found_valid_pair = False
        attempts = 0
        max_attempts = 100
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            a = random.randint(1, c-1)
            b = c - a
            
            max_addend_tokens = 10
            
            if (can_represent_in_n_roman_tokens(a, max_addend_tokens) and 
                can_represent_in_n_roman_tokens(b, max_addend_tokens)):
                input_text, output_text = format_roman(a, b, c)
                
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        if not found_valid_pair:
            continue
    
    return examples

def generate_binary_format(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in binary format."""
    examples = []
    
    while len(examples) < num_examples:
        target_digits = random.randint(min_tokens, max_tokens)
        
        max_sum = 2**target_digits - 1
        min_sum = 2**(target_digits-1)
        
        c = random.randint(min_sum, max_sum)
        
        if not has_exactly_n_binary_digits(c, target_digits):
            continue
            
        found_valid_pair = False
        attempts = 0
        max_attempts = 100
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            a = random.randint(1, c-1)
            b = c - a
            
            max_addend_digits = 10
            
            if (can_represent_in_n_binary_digits(a, max_addend_digits) and 
                can_represent_in_n_binary_digits(b, max_addend_digits)):
                input_text, output_text = format_binary(a, b, c)
                
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        if not found_valid_pair:
            continue
    
    return examples

def generate_binary_format_remapped(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in binary format."""
    examples = []
    
    while len(examples) < num_examples:
        target_digits = random.randint(min_tokens, max_tokens)
        
        max_sum = 2**target_digits - 1
        min_sum = 2**(target_digits-1)
        
        c = random.randint(min_sum, max_sum)
        
        if not has_exactly_n_binary_digits(c, target_digits):
            continue
            
        found_valid_pair = False
        attempts = 0
        max_attempts = 100
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            a = random.randint(1, c-1)
            b = c - a
            
            max_addend_digits = 10
            
            if (can_represent_in_n_binary_digits(a, max_addend_digits) and 
                can_represent_in_n_binary_digits(b, max_addend_digits)):
                input_text, output_text = format_binary(a, b, c)
                input_text, output_text = remap_binary(input_text, output_text)
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        if not found_valid_pair:
            continue
    
    return examples

def generate_hex_format(num_examples, min_tokens=3, max_tokens=10):
    """Generate examples in hexadecimal format."""
    examples = []
    
    while len(examples) < num_examples:
        target_digits = random.randint(min_tokens, max_tokens)
        
        max_sum = 16**target_digits - 1
        min_sum = 16**(target_digits-1)
        
        c = random.randint(min_sum, max_sum)
        
        if not has_exactly_n_hex_digits(c, target_digits):
            continue
            
        found_valid_pair = False
        attempts = 0
        max_attempts = 100
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            a = random.randint(1, c-1)
            b = c - a
            
            max_addend_digits = 10
            
            if (can_represent_in_n_hex_digits(a, max_addend_digits) and 
                can_represent_in_n_hex_digits(b, max_addend_digits)):
                input_text, output_text = format_hex(a, b, c)
                
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        if not found_valid_pair:
            continue
    
    return examples

# ---- Example Registry and Core Functions ----

# Global registry to track examples across different dataset generation calls
REGISTRY = {
    "test": set(),  # Examples in test sets
    "train": set(),  # Examples in train sets
}

def reset_example_registry():
    """Reset the global example registry."""
    global REGISTRY
    REGISTRY = {
        "test": set(),
        "train": set(),
    }

def example_to_tuple(example):
    """Convert an example dict to a hashable tuple."""
    return (example["input"], example["output"])

def create_exact_oversampled_dataset(examples, target_size):
    """
    Create a dataset with exactly target_size examples by oversampling.
    
    Args:
        examples: List of example dictionaries
        target_size: Exact number of examples wanted
    
    Returns:
        List with exactly target_size examples
    """
    num_examples = len(examples)
    if num_examples == 0:
        raise ValueError("Cannot oversample from an empty list of examples")
        
    if num_examples >= target_size:
        # If we already have enough examples, just return the first target_size
        return examples[:target_size]
        
    # Calculate repetitions needed
    full_repetitions = target_size // num_examples
    additional_examples = target_size % num_examples
    
    # Create oversampled dataset
    result = []
    for _ in range(full_repetitions):
        result.extend(examples)
    
    # Add remaining examples if needed
    if additional_examples > 0:
        extra = examples.copy()
        random.shuffle(extra)
        result.extend(extra[:additional_examples])
    
    assert len(result) == target_size, f"Expected {target_size} examples, got {len(result)}"
    return result

def _generate_examples(generator, num_examples, for_test=False, min_tokens=3, max_tokens=10, max_attempts_multiplier=20):
    """
    Generate unique examples with no overlap between test and train sets.
    
    Args:
        generator: Generator function to use
        num_examples: Number of examples to generate
        for_test: If True, generating test examples; train examples otherwise
        min_tokens, max_tokens: Token length range
        max_attempts_multiplier: Controls how hard to try before giving up
    
    Returns:
        List of unique examples
    """
    examples = []
    attempts = 0
    max_attempts = num_examples * max_attempts_multiplier
    
    print(f"Generating {num_examples} {'test' if for_test else 'train'} examples with {generator.__name__}...")
    
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        
        # Generate a single example
        if generator.__name__ in ['generate_original_format', 'generate_roman_format',
                                'generate_binary_format', 'generate_hex_format']:
            example = generator(1, min_tokens=min_tokens, max_tokens=max_tokens)[0]
        else:
            example = generator(1)[0]  # For generators that don't support token range
            
        example_tuple = example_to_tuple(example)
        
        # If generating test examples, just check against existing test examples
        if for_test:
            if example_tuple not in REGISTRY["test"]:
                examples.append(example)
                REGISTRY["test"].add(example_tuple)
        # If generating train examples, check against BOTH test and train to avoid overlap
        else:
            if (example_tuple not in REGISTRY["train"] and 
                example_tuple not in REGISTRY["test"]):  # This is the key check that prevents overlap
                examples.append(example)
                REGISTRY["train"].add(example_tuple)
        
        # Print progress occasionally
        if attempts % 5000 == 0:
            print(f"  Progress: {len(examples)}/{num_examples} examples after {attempts} attempts")
    
    if len(examples) < num_examples:
        print(f"WARNING: Could only generate {len(examples)} unique examples, requested {num_examples}")
    
    return examples

def verify_no_overlap():
    """Check if there is any overlap between test and train sets."""
    overlap = REGISTRY["test"].intersection(REGISTRY["train"])
    if overlap:
        print(f"ERROR: Found {len(overlap)} examples in both train and test sets!")
        print("Sample overlapping examples:")
        for i, ex in enumerate(list(overlap)[:3]):
            print(f"  {i+1}. {ex}")
        return False
    
    print(f"✓ VERIFIED: No overlap between train and test sets.")
    return True

def generate_mixed_dataset(
    output_dir,
    total_train,
    total_test,
    generator_ratios,
    test_generators=None,
    oversample_factor=1,
    random_seed=None,
    min_tokens=3,
    max_tokens=10,
    generate_test_first=True
):
    """
    Generate a mixed dataset with arbitrary generators and ratios.
    Ensures no overlap between test and train sets.
    
    Args:
        output_dir: Directory to save the dataset
        total_train: Total number of training examples
        total_test: Total number of test examples
        generator_ratios: Dict mapping generator functions to their ratios (must sum to 1.0)
        test_generators: List of generators to use for test sets or a single generator.
                         If None, uses the same ratio mix as training.
        oversample_factor: Number of times to repeat generated examples (for oversampling)
        random_seed: Random seed for reproducibility
        min_tokens: Minimum token count for results
        max_tokens: Maximum token count for results
        generate_test_first: If True, generate test set before train set (recommended)
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Validate ratios
    ratio_sum = sum(generator_ratios.values())
    if not 0.999 <= ratio_sum <= 1.001:  # Allow for small floating-point errors
        raise ValueError(f"Generator ratios must sum to 1.0, got {ratio_sum}")
    
    # Create output directory
    create_dirs(output_dir)
    
    # PHASE 1: Generate test examples first (if requested)
    if generate_test_first and total_test > 0:
        test_examples = []
        
        if test_generators is None:
            # Use the same ratio mix as training
            for generator, ratio in generator_ratios.items():
                num_examples = int(total_test * ratio)
                examples = _generate_examples(
                    generator=generator, 
                    num_examples=num_examples, 
                    for_test=True,
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                test_examples.extend(examples)
            
            # Shuffle and save
            random.shuffle(test_examples)
            with open(os.path.join(output_dir, "test.json"), "w") as f:
                json.dump(test_examples, f, indent=2)
            
            print(f"Generated test set: {len(test_examples)} examples")
        else:
            # Handle specified test generators
            if callable(test_generators):
                test_generators = [test_generators]
            
            for i, generator in enumerate(test_generators):
                generator_name = generator.__name__.replace("generate_", "")
                
                test_examples = _generate_examples(
                    generator=generator, 
                    num_examples=total_test,
                    for_test=True,
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                
                # Save to file with appropriate name
                test_filename = f"test_{i}.json"
                with open(os.path.join(output_dir, test_filename), "w") as f:
                    json.dump(test_examples, f, indent=2)
                
                print(f"Generated test set: {len(test_examples)} examples in {test_filename}")
    
    # PHASE 2: Generate training examples
    if total_train > 0:
        train_examples = []
        
        for generator, ratio in generator_ratios.items():
            num_examples = int(total_train * ratio)
            examples = _generate_examples(
                generator=generator, 
                num_examples=num_examples, 
                for_test=False,  # These are train examples
                min_tokens=min_tokens, 
                max_tokens=max_tokens
            )
            train_examples.extend(examples)
        
        # Apply oversampling if requested
        if oversample_factor > 1:
            original_count = len(train_examples)
            target_count = original_count * oversample_factor
            
            print(f"Oversampling training data: {original_count} examples → {target_count} examples")
            train_examples = create_exact_oversampled_dataset(train_examples, target_count)
        
        # Shuffle and save
        random.shuffle(train_examples)
        with open(os.path.join(output_dir, "train.json"), "w") as f:
            json.dump(train_examples, f, indent=2)
        
        print(f"Generated train set: {len(train_examples)} examples")
        
    # PHASE 3: Generate test examples if they weren't generated first
    if not generate_test_first and total_test > 0:
        print("WARNING: Generating test set after train set is not recommended!")
        
        # The implementation is the same as PHASE 1, just in a different order
        test_examples = []
        
        if test_generators is None:
            # Use the same ratio mix as training
            for generator, ratio in generator_ratios.items():
                num_examples = int(total_test * ratio)
                examples = _generate_examples(
                    generator=generator, 
                    num_examples=num_examples, 
                    for_test=True,
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                test_examples.extend(examples)
            
            # Shuffle and save
            random.shuffle(test_examples)
            with open(os.path.join(output_dir, "test.json"), "w") as f:
                json.dump(test_examples, f, indent=2)
            
            print(f"Generated test set: {len(test_examples)} examples")
        else:
            # Handle specified test generators
            if callable(test_generators):
                test_generators = [test_generators]
            
            for generator in test_generators:
                generator_name = generator.__name__.replace("generate_", "")
                
                test_examples = _generate_examples(
                    generator=generator, 
                    num_examples=total_test,
                    for_test=True,
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                
                # Save to file with appropriate name
                test_filename = f"test_{generator_name}.json"
                with open(os.path.join(output_dir, test_filename), "w") as f:
                    json.dump(test_examples, f, indent=2)
                
                print(f"Generated test set: {len(test_examples)} examples in {test_filename}")
    
    # Print statistics
    print(f"\nDataset generation complete:")
    print(f"  - Training examples: {len(REGISTRY['train'])}")
    print(f"  - Test examples: {len(REGISTRY['test'])}")
    
    # Verify no overlap
    verify_no_overlap()