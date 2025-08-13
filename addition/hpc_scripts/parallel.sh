#!/bin/bash

# Define the data lengths to iterate through
DATA_LENGTHS=(512 1024 2048 4096 8192 12800 16384 25600)

# Loop through each data length and submit a separate job
for length in "${DATA_LENGTHS[@]}"; do
    # Submit a job for this specific length
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_${length}                    # Job name includes length
#SBATCH --output=logs/train/training_${length}_%j.out # Log file includes length
#SBATCH --error=logs/train/training_${length}_%j.err  # Error file includes length
#SBATCH --time=12:00:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                     # Number of nodes requested
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --gpus=1                                      # Number of GPUs requested
#SBATCH --cpus-per-task=16                            # Number of CPU cores per task
#SBATCH --mem=64GB                                    # Memory limit
#SBATCH --partition=small-g                           # Partition name

echo "Starting training with data.length=${length}"

# Run the training with this specific length parameter
singularity exec \\
    ${SIF} \\
    python train.py data.length=${length}

echo "Completed training with data.length=${length}"
EOF

    echo "Submitted job for data.length=${length}"
done

echo "All jobs have been submitted and will run in parallel (subject to cluster availability)"