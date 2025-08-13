import json
import os
import glob
import random
from pathlib import Path

def process_json_file(input_file, output_file, folder_letter):
    """
    Process a single JSON file, transforming its structure and prepending the folder letter.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the output JSON file.
        folder_letter (str): Letter to prepend to each input (A, B, or C).
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Transform the data structure
    transformed_data = []
    for item in data:
        text = item.get("text", "")
        
        # Split the text by "Command:" to separate input and output
        parts = text.split(" Command: ")
        
        if len(parts) == 2:
            input_text, output_text = parts
            
            # Prepend the folder letter to the input
            input_text = f"{folder_letter} {input_text}"
            
            transformed_item = {
                "input": input_text,
                "output": output_text
            }
            
            transformed_data.append(transformed_item)
    
    # Save the transformed data to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    return transformed_data

def process_all_files():
    """
    Process all JSON files from folders A, B, and C and save transformed files
    to t_search_post folder, maintaining the original filename but not the folder structure.
    """
    base_input_dir = "data/t_search"
    base_output_dir = "data/t_search_post"
    
    # Make sure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process files from each folder (A, B, C)
    for folder_letter in ['A', 'B', 'C']:
        input_folder = os.path.join(base_input_dir, folder_letter)
        output_folder = os.path.join(base_output_dir, folder_letter)
        
        # Make sure the output subfolder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all JSON files in the folder
        json_files = glob.glob(os.path.join(input_folder, "*.json"))
        
        for input_file in json_files:
            # Create output file path - just the filename, not the folder structure
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_folder, filename)
            
            # Process the file
            process_json_file(input_file, output_file, folder_letter)

def merge_into_final_files():
    """
    Merge existing train.json and test.json files from folders A, B, C into merged folder
    """
    base_dir = "data/t_search_post"
    merged_dir = os.path.join(base_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Combine train.json files
    train_data = []
    for folder_letter in ['A', 'B', 'C']:
        train_file = os.path.join(base_dir, folder_letter, "train.json")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                data = json.load(f)
                train_data.extend(data)
    
    # Shuffle train data
    random.shuffle(train_data)
    
    # Save merged train.json
    final_train_file = os.path.join(merged_dir, "train.json")
    with open(final_train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Created merged train file with {len(train_data)} examples at {final_train_file}")
    
    # Combine test.json files
    test_data = []
    for folder_letter in ['A', 'B', 'C']:
        test_file = os.path.join(base_dir, folder_letter, "test.json")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                data = json.load(f)
                test_data.extend(data)
    
    # Save merged test.json
    final_test_file = os.path.join(merged_dir, "test.json")
    with open(final_test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Created merged test file with {len(test_data)} examples at {final_test_file}")
    
    return train_data, test_data

if __name__ == "__main__":
    print("Processing all files...")
    process_all_files()
    
    print("\nMerging train/test files into final merged files...")
    merge_into_final_files()
    
    print("\nDone!")