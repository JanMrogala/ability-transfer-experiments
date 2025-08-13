import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config")
def preprocess(cfg: DictConfig):
    print("Processing Czech files to add <|endoftext|> at the end of each story...")
    
    # Paths for Czech files
    cs_files = {
        "train": f"{cfg.data.datapath}/cs/train.txt",
        "val": f"{cfg.data.datapath}/cs/val.txt"
    }
    
    for file_type, file_path in cs_files.items():
        if os.path.exists(file_path):
            print(f"Processing Czech {file_type} file: {file_path}")
            
            # Read stories (one per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                stories = [line.strip() for line in f.readlines() if line.strip()]
            
            # Write with each story on its own line followed by <|endoftext|>
            with open(file_path, 'w', encoding='utf-8') as f:
                for story in stories:
                    f.write(f"{story}\n<|endoftext|>\n")
            
            print(f"Processed {len(stories)} Czech stories in {file_type}")
    
    print("Processing complete.")

if __name__ == "__main__":
    preprocess()