#!/bin/bash

# Define the data formats and their corresponding tokenizer formats
declare -A TOK_FORMATS=(
  ["binary_addition"]="addition_binary"
  ["addition_binary"]="addition_binary"
  ["hex_addition"]="addition_hex"
  ["addition_hex"]="addition_hex"
  ["roman_addition"]="addition_roman"
  ["addition_roman"]="addition_roman"
  ["letter_addition"]="addition_letter"
  ["addition_letter"]="addition_letter"
)

# Define all data formats to iterate through
DATA_FORMATS=("binary_addition" "addition_binary" "hex_addition" "addition_hex" "roman_addition" "addition_roman" "addition_letter" "letter_addition")

# Loop through each data format and submit a separate job
for format in "${DATA_FORMATS[@]}"; do
    # Get the corresponding tokenizer format
    tok_format="${TOK_FORMATS[$format]}"
    
    # Submit a job for this specific format
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_${format}                    # Job name includes format
#SBATCH --output=logs/train/training_${format}_%j.out # Log file includes format
#SBATCH --error=logs/train/training_${format}_%j.err  # Error file includes format
#SBATCH --time=12:00:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                     # Number of nodes requested
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --gpus=1                                      # Number of GPUs requested
#SBATCH --cpus-per-task=16                            # Number of CPU cores per task
#SBATCH --mem=64GB                                    # Memory limit
#SBATCH --partition=small-g                           # Partition name

echo "Starting training with data.format=${format} and data.tok_format=${tok_format}"
# Run the training with this specific format parameter
singularity exec \
    \${SIF} \
    python train.py data.format=${format} data.tok_format=${tok_format}

echo "Completed training with data.format=${format} and data.tok_format=${tok_format}"
EOF

    echo "Submitted job for data.format=${format} and data.tok_format=${tok_format}"
done

echo "All jobs have been submitted and will run in parallel (subject to cluster availability)"