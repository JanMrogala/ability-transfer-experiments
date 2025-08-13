import json
import os
import random
from tqdm import tqdm

def generate_examples(num_examples):
    examples = []
    while len(examples) < num_examples:
        # Generate random numbers
        a = random.randint(1, 99999)
        b = random.randint(1, 99999)
        c = a + b  
        
        # Only include examples where the sum has exactly 5 digits
        if len(str(c)) == 5:
            # Convert to space-separated digits
            a_spaced = ' '.join(str(a))
            b_spaced = ' '.join(str(b))
            c_spaced = ' '.join(str(c))
            
            input_text = f"{a_spaced} + {b_spaced}"
            output_text = f"{c_spaced}"
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples

def main():

    nums = [500, 1000, 3000, 5000]
    
    # Generate training data
    for num in tqdm(nums):
        train_examples = generate_examples(num)
        with open(f"data/train{num}.json", "w") as f:
            json.dump(train_examples, f, indent=2)
        print(f"Generated {len(train_examples)} training examples")
    
    # Generate test data
    test_examples = generate_examples(2048)
    with open(f"data/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)

    print(f"Generated {len(test_examples)} test examples")

if __name__ == "__main__":
    main()