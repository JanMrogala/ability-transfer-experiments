import json
import os
import random
from tqdm import tqdm

# Define the mapping of digits to letters
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

def convert_to_letters(num_str):
    """Convert a string of digits to letters."""
    return ' '.join(DIGIT_TO_LETTER[digit] for digit in num_str)

def format_original(a, b, c):
    """Format addition in original format: 1 2 3 4 5 + 1 2 3 4 5 = 2 4 6 9 0"""
    a_spaced = ' '.join(str(a))
    b_spaced = ' '.join(str(b))
    c_spaced = ' '.join(str(c))
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_new(a, b, c):
    """Format addition in new format: ! A B C D E , A B C D E . C E G J A"""
    a_letters = convert_to_letters(str(a))
    b_letters = convert_to_letters(str(b))
    c_letters = convert_to_letters(str(c))
    
    input_text = f"! {a_letters} , {b_letters}"
    output_text = f"{c_letters}"
    
    return input_text, output_text

def format_mixed_odd_even(a, b, c):
    """Format where odd digits use new language, even digits use original.
    The structure follows the new language format (with ! and .)."""
    # Convert each digit: odd → letter, even → digit
    a_mixed = []
    for digit in str(a):
        if int(digit) % 2 == 1:  # Odd digit
            a_mixed.append(DIGIT_TO_LETTER[digit])
        else:  # Even digit
            a_mixed.append(digit)
    
    b_mixed = []
    for digit in str(b):
        if int(digit) % 2 == 1:  # Odd digit
            b_mixed.append(DIGIT_TO_LETTER[digit])
        else:  # Even digit
            b_mixed.append(digit)
    
    c_mixed = []
    for digit in str(c):
        if int(digit) % 2 == 1:  # Odd digit
            c_mixed.append(DIGIT_TO_LETTER[digit])
        else:  # Even digit
            c_mixed.append(digit)
    
    # Join with spaces
    a_mixed_str = ' '.join(a_mixed)
    b_mixed_str = ' '.join(b_mixed)
    c_mixed_str = ' '.join(c_mixed)
    
    # Use new structure (!)
    input_text = f"! {a_mixed_str} , {b_mixed_str}"
    output_text = f"{c_mixed_str}"
    
    return input_text, output_text

def format_old_right_new_left(a, b, c):
    """Format with new language on left side (operands) and original language on right side (result)."""
    # Left side (operands) always uses new language
    a_letters = convert_to_letters(str(a))
    b_letters = convert_to_letters(str(b))
    
    # Right side (result) uses original language
    c_formatted = ' '.join(str(c))  # Original language
    
    # Use new structure (!)
    input_text = f"! {a_letters} , {b_letters}"
    output_text = f"{c_formatted}"
    
    return input_text, output_text

def format_old_left_new_right(a, b, c):
    """Format with original language on left side (operands) and new language on right side (result)."""
    # Left side (operands) uses original language
    a_formatted = ' '.join(str(a))  # Original language
    b_formatted = ' '.join(str(b))  # Original language
    
    # Right side (result) uses new language
    c_letters = convert_to_letters(str(c))
    
    # Use original structure (+)
    input_text = f"{a_formatted} + {b_formatted}"
    output_text = f"{c_letters}"
    
    return input_text, output_text

def format_mixed_ratio(a, b, c, token_change_ratio=0.5):
    """Format with a specified ratio of tokens changed to new language.
    token_change_ratio controls what fraction of digits get converted to letters."""
    # Convert digits to lists for manipulation
    a_digits = list(str(a))
    b_digits = list(str(b))
    c_digits = list(str(c))
    
    # Determine which positions to convert based on the token_change_ratio
    all_positions = list(range(len(a_digits) + len(b_digits) + len(c_digits)))
    num_to_change = int(len(all_positions) * token_change_ratio)
    positions_to_change = random.sample(all_positions, num_to_change)
    
    # Convert selected positions
    a_mixed = []
    for i, digit in enumerate(a_digits):
        if i in positions_to_change:
            a_mixed.append(DIGIT_TO_LETTER[digit])
        else:
            a_mixed.append(digit)
    
    b_mixed = []
    for i, digit in enumerate(b_digits):
        if i + len(a_digits) in positions_to_change:
            b_mixed.append(DIGIT_TO_LETTER[digit])
        else:
            b_mixed.append(digit)
    
    c_mixed = []
    for i, digit in enumerate(c_digits):
        if i + len(a_digits) + len(b_digits) in positions_to_change:
            c_mixed.append(DIGIT_TO_LETTER[digit])
        else:
            c_mixed.append(digit)
    
    # Join with spaces
    a_mixed_str = ' '.join(a_mixed)
    b_mixed_str = ' '.join(b_mixed)
    c_mixed_str = ' '.join(c_mixed)
    
    # Use new structure (!) if any part uses new language, otherwise use original
    has_new_language = any(d in DIGIT_TO_LETTER.values() for d in a_mixed + b_mixed + c_mixed)
    if has_new_language:
        input_text = f"! {a_mixed_str} , {b_mixed_str}"
    else:
        input_text = f"{a_mixed_str} + {b_mixed_str}"
    
    output_text = f"{c_mixed_str}"
    
    return input_text, output_text

def generate_examples(num_examples, format_func, **kwargs):
    """Generate examples using the specified formatting function."""
    examples = []
    while len(examples) < num_examples:
        # Generate random numbers
        a = random.randint(1, 99999)
        b = random.randint(1, 99999)
        c = a + b
        
        # Only include examples where the sum has exactly 5 digits
        if len(str(c)) == 5:
            input_text, output_text = format_func(a, b, c, **kwargs)
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples

def generate_mix(num_train, num_test, new_ratio=1.00):
    """Generate mixed dataset with specified ratio of new format."""
    # Create directory
    create_dirs("data/new_lang_10k")
    
    # Generate training data - mix of original and new
    train_examples = []
    num_new = int(num_train * new_ratio)
    num_original = num_train - num_new
    
    # Generate original format examples
    train_examples.extend(generate_examples(num_original, format_original))
    
    # Generate new format examples
    ex = generate_examples(num_new, format_new)
    exs = ex*23
    train_examples.extend(exs)
    
    # Shuffle examples
    random.shuffle(train_examples)
    
    # Write training data
    with open("data/new_lang_10k/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test data - only new format (5% modified examples)
    test_examples = generate_examples(num_test, format_new)
    with open("data/new_lang_10k/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated mix_only: {len(train_examples)} train ({num_new} new format, {num_original} original), {len(test_examples)} test (all new format)")

def generate_odd_vs_even(num_train, num_test, modified_ratio=0.05):
    """Generate dataset with 95% original language and 5% using odd/even representation."""
    # Create directory
    create_dirs("data/odd_vs_even")
    
    # Generate training data - mix of 95% original and 5% odd/even format
    train_examples = []
    num_modified = int(num_train * modified_ratio)
    num_original = num_train - num_modified
    
    # Generate original format examples
    train_examples.extend(generate_examples(num_original, format_original))
    
    # Generate odd/even format examples
    train_examples.extend(generate_examples(num_modified, format_mixed_odd_even))
    
    # Shuffle examples
    random.shuffle(train_examples)
    
    # Write training data
    with open("data/odd_vs_even/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test data - all odd/even format
    test_examples = generate_examples(num_test, format_mixed_odd_even)
    with open("data/odd_vs_even/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated odd_vs_even: {len(train_examples)} train ({num_modified} odd/even format, {num_original} original), {len(test_examples)} test (all odd/even format)")

def generate_old_right_vs_new_left(num_train, num_test, modified_ratio=0.05):
    """Generate dataset with 95% original language and 5% with new language on left side."""
    # Create directory
    create_dirs("data/old_right_vs_new_left")
    
    # Generate training data - mix of 95% original and 5% modified format
    train_examples = []
    num_modified = int(num_train * modified_ratio)
    num_original = num_train - num_modified
    
    # Generate original format examples
    train_examples.extend(generate_examples(num_original, format_original))
    
    # Generate new-left format examples
    train_examples.extend(generate_examples(num_modified, format_old_right_new_left))
    
    # Shuffle examples
    random.shuffle(train_examples)
    
    # Write training data
    with open("data/old_right_vs_new_left/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test data - all new-left format
    test_examples = generate_examples(num_test, format_old_right_new_left)
    with open("data/old_right_vs_new_left/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated old_right_vs_new_left: {len(train_examples)} train ({num_modified} new-left format, {num_original} original), {len(test_examples)} test (all new-left format)")

def generate_old_left_vs_new_right(num_train, num_test, modified_ratio=0.05):
    """Generate dataset with 95% original language and 5% with new language on right side."""
    # Create directory
    create_dirs("data/old_left_vs_new_right")
    
    # Generate training data - mix of 95% original and 5% modified format
    train_examples = []
    num_modified = int(num_train * modified_ratio)
    num_original = num_train - num_modified
    
    # Generate original format examples
    train_examples.extend(generate_examples(num_original, format_original))
    
    # Generate new-right format examples
    train_examples.extend(generate_examples(num_modified, format_old_left_new_right))
    
    # Shuffle examples
    random.shuffle(train_examples)
    
    # Write training data
    with open("data/old_left_vs_new_right/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test data - all new-right format
    test_examples = generate_examples(num_test, format_old_left_new_right)
    with open("data/old_left_vs_new_right/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated old_left_vs_new_right: {len(train_examples)} train ({num_modified} new-right format, {num_original} original), {len(test_examples)} test (all new-right format)")

def generate_mixed_ratio(num_train, num_test, modified_ratio=0.05, token_ratios=[0.3, 0.5, 0.7]):
    """Generate dataset with 95% original language and 5% with mixed ratio of token changes."""
    # Create directory
    create_dirs("data/mixed_ratio")
    
    for ratio_idx, token_change_ratio in enumerate(token_ratios):
        # Generate training data - mix of 95% original and 5% mixed format
        train_examples = []
        num_modified = int(num_train * modified_ratio)
        num_original = num_train - num_modified
        
        # Generate original format examples
        train_examples.extend(generate_examples(num_original, format_original))
        
        # Generate mixed ratio format examples
        train_examples.extend(generate_examples(num_modified, format_mixed_ratio, token_change_ratio=token_change_ratio))
        
        # Shuffle examples
        random.shuffle(train_examples)
        
        # Write training data
        with open(f"data/mixed_ratio/train_ratio_{int(token_change_ratio*100)}.json", "w") as f:
            json.dump(train_examples, f, indent=2)
        
        # Generate test data - all mixed ratio format
        test_examples = generate_examples(num_test, format_mixed_ratio, token_change_ratio=token_change_ratio)
        with open(f"data/mixed_ratio/test_ratio_{int(token_change_ratio*100)}.json", "w") as f:
            json.dump(test_examples, f, indent=2)
        
        print(f"Generated mixed_ratio {int(token_change_ratio*100)}%: {len(train_examples)} train ({num_modified} mixed format with {int(token_change_ratio*100)}% token changes, {num_original} original), {len(test_examples)} test (all mixed format)")

def main():
    # Number of examples for each dataset
    num_train = 512  
    num_test = 512     
    modified_ratio = 0.05  # 5% of examples are modified
    
    print("Generating datasets...")
    
    # Variant 1: Mix of formats, with 5% new format in training, 100% new format in testing
    generate_mix(num_train, num_test)
    
    # # Variant 2: Odd vs even digits representation
    # generate_odd_vs_even(num_train, num_test)
    
    # # Variant 3: Original language for numbers after "." with different splits
    # generate_old_right_vs_new_left(num_train, num_test)
    
    # # Variant 4: Original language for numbers before "." with different splits
    # generate_old_left_vs_new_right(num_train, num_test)
    
    # # Variant 5: Mixed ratios of original to new language
    # generate_mixed_ratio(num_train, num_test)
    
    # print("All datasets generated successfully!")

if __name__ == "__main__":
    # Create the base data directory
    create_dirs("data")
    main()