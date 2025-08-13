import json
import os
import random

def create_dirs(*dirs):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def int_to_roman_tokens(num):
    """Convert an integer to a list of Roman numeral tokens."""
    # Map of values to Roman numerals
    val_to_roman = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
        (1, "I")
    ]
    
    roman_tokens = []
    i = 0
    
    while num > 0 and i < len(val_to_roman):
        symbol = val_to_roman[i][1]
        value = val_to_roman[i][0]
        
        if num >= value:
            roman_tokens.append(symbol)
            num -= value
        else:
            i += 1
    
    return roman_tokens

def can_represent_in_five_tokens(num):
    """Check if a number can be represented with 5 or fewer Roman numeral tokens."""
    return len(int_to_roman_tokens(num)) <= 5

def format_original(a, b, c):
    """Format addition in original format: 1 2 3 4 5 + 1 2 3 4 5 = 2 4 6 9 0"""
    a_spaced = ' '.join(str(a))
    b_spaced = ' '.join(str(b))
    c_spaced = ' '.join(str(c))
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_roman(a, b, c):
    """Format addition in Roman numeral format with spaces between tokens"""
    # Convert to Roman tokens
    a_roman = int_to_roman_tokens(a)
    b_roman = int_to_roman_tokens(b)
    c_roman = int_to_roman_tokens(c)
    
    # Join with spaces
    a_spaced = ' '.join(a_roman)
    b_spaced = ' '.join(b_roman)
    c_spaced = ' '.join(c_roman)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def generate_examples_original(num_examples):
    """Generate examples in original format where:
       - Addends can have 1-5 digits each
       - Result must have exactly 5 digits
       - Arithmetic is correct
    """
    examples = []
    
    while len(examples) < num_examples:
        # Generate addends with 1-5 digits each
        a_digits = random.randint(1, 5)
        b_digits = random.randint(1, 5)
        
        # Generate the actual numbers with the specified number of digits
        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 1  # Minimum is 1, not 0
        a_max = (10 ** a_digits) - 1
        a = random.randint(a_min, a_max)
        
        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 1  # Minimum is 1, not 0
        b_max = (10 ** b_digits) - 1
        b = random.randint(b_min, b_max)
        
        # Calculate their sum (ensuring arithmetic correctness)
        c = a + b
        
        # Check if the sum has exactly 5 digits
        if 10000 <= c <= 99999:
            input_text, output_text = format_original(a, b, c)
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples

def has_exactly_five_roman_tokens(num):
    """Check if a number can be represented with exactly 5 Roman numeral tokens."""
    tokens = int_to_roman_tokens(num)
    return len(tokens) == 5

def generate_examples_roman(num_examples):
    """Generate examples in Roman format where:
       - Addends can have 1-5 tokens each
       - Result must have exactly 5 tokens
       - Arithmetic is correct
    """
    examples = []
    
    # Find numbers that can be represented with exactly 5 Roman tokens
    # This improves efficiency by precomputing possible sums
    possible_sums = []
    for i in range(1, 10000):
        if has_exactly_five_roman_tokens(i):
            possible_sums.append(i)
    
    if not possible_sums:
        raise ValueError("Could not find any numbers representable with exactly 5 Roman tokens")
    
    print(f"Found {len(possible_sums)} numbers representable with exactly 5 Roman tokens")
    
    while len(examples) < num_examples:
        # Select a number that can be represented with exactly 5 Roman tokens
        c = random.choice(possible_sums)
        
        # Try to find addends that sum to c
        found_valid_pair = False
        attempts = 0
        max_attempts = 100  # Increased attempts for better success rate
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            # Random split of c into a and b
            a = random.randint(1, c-1)
            b = c - a
            
            # Check if both a and b can be represented with 5 or fewer Roman tokens
            if (can_represent_in_five_tokens(a) and can_represent_in_five_tokens(b)):
                input_text, output_text = format_roman(a, b, c)
                
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        # If we couldn't find a valid pair after many attempts, try a different sum
        if not found_valid_pair:
            continue
    
    return examples

def generate_mix(num_train, num_test, num_roman_train):
    """Generate mixed dataset with specified number of Roman numeral examples."""
    # Create directory
    create_dirs("data/roman_numeral_full")
    
    # Generate Roman numeral examples for test set
    print(f"Generating {num_test} test examples (all Roman numeral format)...")
    test_examples = generate_examples_roman(num_test)
    
    # Generate Roman numeral examples for training set
    print(f"Generating {num_roman_train} Roman numeral format examples for training...")
    roman_train_examples = generate_examples_roman(num_roman_train)
    
    # Generate original format examples for the rest of the training set
    num_original = num_train - num_roman_train
    print(f"Generating {num_original} original format examples...")
    original_train_examples = generate_examples_original(num_original)
    
    # Combine and shuffle training examples
    train_examples = original_train_examples + roman_train_examples
    random.shuffle(train_examples)
    
    # Write datasets to files
    with open("data/roman_numeral_full/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    with open("data/roman_numeral_full/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated mix: {len(train_examples)} train ({num_roman_train} Roman format, {num_original} original), {len(test_examples)} test (all Roman format)")

def main():
    # Number of examples for each dataset
    num_train = 0
    num_test = 512
    num_roman_train = 12228  # Exact number of Roman examples in training set
    
    print("Generating Roman numeral arithmetic dataset...")
    
    # Generate datasets
    generate_mix(num_train, num_test, num_roman_train)
    
    print("Dataset generated successfully!")

if __name__ == "__main__":
    # Create the base data directory
    create_dirs("data")
    main()