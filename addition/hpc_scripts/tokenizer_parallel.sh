#!/bin/bash
#SBATCH --job-name=tokenizer                            # Job name
#SBATCH --output=logs/tokenizer/tokenizer_%j.out        # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/tokenizer/tokenizer_%j.err         # Error log
#SBATCH --time=00:20:00                                 # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                       # Number of nodes requested
#SBATCH --ntasks=1                                      # Number of tasks (processes)
#SBATCH --gpus=1                                        # Number of GPUs requested
#SBATCH --cpus-per-task=16                              # Number of CPU cores per task
#SBATCH --mem=64GB                                      # Memory limit
#SBATCH --partition=small-g                             # Partition name

# Define the data formats to iterate through - NO COMMAS between array elements
DATA_FORMATS=("binary_addition" "addition_binary" "hex_addition" "addition_hex" "roman_addition" "addition_roman")

# Loop through each data format and submit a separate job
for format in "${DATA_FORMATS[@]}"; do
    # Submit a job for this specific format
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tok_${format}                    # Job name includes format
#SBATCH --output=logs/train/training_${format}_%j.out # Log file includes format
#SBATCH --error=logs/train/training_${format}_%j.err  # Error file includes format
#SBATCH --time=00:05:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                     # Number of nodes requested
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --gpus=1                                      # Number of GPUs requested
#SBATCH --cpus-per-task=16                            # Number of CPU cores per task
#SBATCH --mem=64GB                                    # Memory limit
#SBATCH --partition=small-g                           # Partition name

echo "Creating tokenizer with data.format=${format}"
# For Hydra, make sure to properly quote or escape special characters
singularity exec \
    \${SIF} \
    python utils/create_tokenizer.py "data.format=${format}"
EOF
    echo "Submitted job for data.format=${format}"
done

echo "All jobs have been submitted and will run in parallel (subject to cluster availability)"