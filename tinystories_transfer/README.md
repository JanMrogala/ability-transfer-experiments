# Multilingual Language Model Training Framework

A comprehensive framework for training small-scale language models in multiple languages using PyTorch Lightning and Hugging Face Transformers. This project provides a complete pipeline for preprocessing data, building tokenizers, tokenizing datasets, and training GPTNeo language models with performance evaluation.

## Features

- 🌐 **Multilingual Support**: Train models on different languages (currently English and Czech)
- 🔄 **Complete Pipeline**: From raw text preprocessing to model training and evaluation
- 🚅 **High-Performance Training**: PyTorch Lightning for efficient, distributed training
- 📊 **Integrated Monitoring**: WandB integration for experiment tracking
- 📝 **Text Generation**: Automatic text generation samples during training
- 📏 **Perplexity Evaluation**: Built-in perplexity calculation for model assessment
- 🖥️ **HPC Support**: Ready-to-use scripts for high-performance computing environments

## Project Structure

```
├── configs/
│   └── config.yaml           # Hydra configuration file
├── data/                     # Data directory (gitignored)
│   ├── en/                   # English dataset
│   └── cs/                   # Czech dataset
├── hpc_scripts/              # Slurm scripts for HPC environments
│   ├── preprocess.sh         # Preprocessing script
│   ├── tokenizer.sh          # Tokenizer creation script
│   ├── tokenize_data.sh      # Data tokenization script
│   └── train.sh              # Training script
├── models/                   # Saved models directory (gitignored)
├── tokenizers/               # Saved tokenizers directory
├── utils/
│   ├── preprocess.py         # Data preprocessing utility
│   ├── tokenizer.py          # Tokenizer creation utility
│   └── tokenize_data.py      # Data tokenization utility
├── train.py                  # Main training script
├── ppl.py                    # Perplexity calculation implementation
├── requirements.txt          # Project dependencies
└── .gitignore                # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pe-hy/tinystories_transfer.git
cd tinystories_transfer
```

2. Create an environment and install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Organize your text data in the following structure:
```
data/
├── en/
│   ├── train.txt
│   └── val.txt
└── cs/
    ├── train.txt
    └── val.txt
```

2. Preprocess the data:
```bash
python utils/preprocess.py
```

### Tokenizer Creation

Create BPE tokenizers for your languages:
```bash
python utils/tokenizer.py
```

### Data Tokenization

Tokenize your text data with the created tokenizers:
```bash
python utils/tokenize_data.py
```

### Model Training

Train the model:
```bash
python train.py
```

### HPC Usage

For high-performance computing environments with Slurm:

1. Preprocessing:
```bash
sbatch hpc_scripts/preprocess.sh
```

2. Create tokenizers:
```bash
sbatch hpc_scripts/tokenizer.sh
```

3. Tokenize data:
```bash
sbatch hpc_scripts/tokenize_data.sh
```

4. Training:
```bash
sbatch hpc_scripts/train.sh
```

## Configuration

The project uses Hydra for configuration management. Key configuration parameters in `configs/config.yaml`:

### Data Configuration
```yaml
data:
  datapath: "data"
  languages: ["en", "cs"]
  # ... more options
```

### Model Configuration
```yaml
model:
  batch_size: 80
  accumulate_grad_batches: 16
  block_size: 512
  epochs: 50
  n_head: 16
  hidden_size: 768
  # ... more options
```

### Tokenizer Configuration
```yaml
tokenizer:
  vocab_size: 30000
  # ... more options
```

## Features in Detail

### Text Generation During Training

The model automatically generates sample texts during training based on prompts in configured languages. These samples are logged to WandB for qualitative assessment.

### Perplexity Evaluation

The framework includes comprehensive perplexity calculation for model evaluation, allowing you to assess language model quality beyond just loss values.

### Weights & Biases Integration

Training progress, hyperparameters, generated text samples, and evaluation metrics are logged to Weights & Biases for easy experiment tracking.

## Customization

### Adding New Languages

To add a new language:
1. Add the language code to the `languages` list in `configs/config.yaml`
2. Create data folders for the new language
3. Run the preprocessing, tokenizer creation, and data tokenization steps

### Modifying Model Architecture

Model architecture parameters can be adjusted in the `model` section of the configuration file:
```yaml
model:
  n_head: 16          # Number of attention heads
  hidden_size: 768    # Hidden size dimension
  n_layer: 4          # Number of transformer layers
  # ... more options
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project leverages several powerful libraries:
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hydra](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)
