"""
Main training script for language model training using PyTorch Lightning and Hydra.
This script assumes tokenization has already been performed by tokenize_data.py.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    GPT2TokenizerFast,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    DataCollatorForLanguageModeling
)
import html
# First, import the Perplexity class at the top of your train.py file
from ppl import Perplexity  # assuming you've renamed the file to perplexity.py

class TinyStoriesModule(pl.LightningModule):
    """
    PyTorch Lightning module for TinyStories language model.
    """
    def __init__(self, cfg, tokenizer, train_dataset=None, val_dataset=None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Create model
        self.model = self._create_model()
        # Disable cache during training
        self.model.config.use_cache = False

        # For accumulating validation texts
        self.validation_texts = []
        
        # Flag to print examples only once
        self.examples_printed = False

    def _create_model(self):
        """Create a GPTNeo model."""
        config = GPTNeoConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=self.cfg.model.max_position_embeddings,
            hidden_size=self.cfg.model.hidden_size,
            num_layers=self.cfg.model.n_layer,
            attention_types=[[['global', 'local'], 2]],
            num_heads=self.cfg.model.n_head,
            intermediate_size=None,  # careful
            window_size=256,
            activation_function='gelu_new',
            resid_dropout=0,
            embed_dropout=0,
            attention_dropout=0,
            classifier_dropout=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return GPTNeoForCausalLM(config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def validation_step(self, batch, batch_idx):
        # Standard validation loss calculation
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        
        # For perplexity calculation, collect the input texts
        input_ids = batch["input_ids"]
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Store all texts for later perplexity calculation
        self.validation_texts.extend(texts)
        
        return loss

    def on_validation_epoch_end(self):
        device_str = "cuda" if str(self.device).startswith("cuda") else "cpu"
        # Calculate perplexity on all collected validation texts
        ppl_calculator = Perplexity(self.model, self.tokenizer, device=device_str)
        
        # To avoid memory issues with very large validation sets,
        # we can limit the number of texts used for perplexity calculation
        max_texts = self.cfg.data.sampling.num_val  # Adjust based on your memory constraints
        texts_for_ppl = self.validation_texts[:max_texts] if len(self.validation_texts) > max_texts else self.validation_texts
        
        # Calculate perplexity on the collected texts
        ppl_results = ppl_calculator._compute(texts_for_ppl, add_start_token=False)
        
        # Log the perplexity metrics
        self.log('val_perplexity', ppl_results["mean_perplexity"])
        self.log('val_token_loss', ppl_results["mean_per_token_loss"])

        # Clear the texts for the next epoch
        self.validation_texts = []

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Generate and log text samples only once per full training epoch
        self.generate_and_log_samples(epoch=self.current_epoch)
        
    def print_random_examples(self):
        """Print 100 random examples from train and val datasets."""
        if self.examples_printed:
            return
            
        print("\n" + "="*80)
        print("DATASET EXAMPLES")
        print("="*80)
        
        # Print train examples
        if self.train_dataset is not None:
            print(f"\n--- TRAIN SET EXAMPLES (100 random from {len(self.train_dataset)}) ---")
            train_indices = torch.randperm(len(self.train_dataset))[:100].tolist()
            for i, idx in enumerate(train_indices[:10]):  # Show first 10 for readability
                example = self.train_dataset[idx]
                text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
                print(f"Train {i+1}: {text[:200]}{'...' if len(text) > 200 else ''}")
                
        # Print val examples  
        if self.val_dataset is not None:
            print(f"\n--- VAL SET EXAMPLES (100 random from {len(self.val_dataset)}) ---")
            val_indices = torch.randperm(len(self.val_dataset))[:100].tolist()
            for i, idx in enumerate(val_indices[:10]):  # Show first 10 for readability
                example = self.val_dataset[idx]
                text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
                print(f"Val {i+1}: {text[:200]}{'...' if len(text) > 200 else ''}")
                
        print("="*80)
        self.examples_printed = True

    def training_step(self, batch, batch_idx):
        # Print examples on first training step
        if batch_idx == 0 and not self.examples_printed:
            self.print_random_examples()
            
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def generate_and_log_samples(self, epoch):
        """Generate text samples based on the current training language and log them to WandB."""
        # Get model ready for generation
        self.model.eval()
        
        # Detect the current language from config
        current_lang = self.cfg.tokenized_data.tok
        
        print(f"\n{'='*60}")
        print(f"GENERATING SAMPLES - EPOCH {epoch} - LANGUAGE: {current_lang}")
        print(f"{'='*60}")
        
        # Set up generation parameters
        generation_config = {
            "max_length": 100,              # Length of generated text
            "num_return_sequences": 1,      # Generate one text per prompt
            "do_sample": True,              # Use sampling (not greedy decoding)
            "top_p": 0.95,                  # Nucleus sampling
            "top_k": 50,                    # Top-k sampling
            "temperature": 0.8,             # Temperature for sampling
            "no_repeat_ngram_size": 2,      # Avoid repeating 2-grams
            "pad_token_id": self.tokenizer.eos_token_id,  # Set pad token
        }
        
        # Define prompts based on the language
        if current_lang == "en":
            prompts = [
                "Once upon a time",
                "In a small village",
                "The little girl",
                "A funny thing happened",
                "The big dog",
                "Today I learned",
                "My favorite toy",
                "The best day ever",
                "When I grow up",
                "The magical forest"
            ]
        elif current_lang == "cs":
            prompts = [
                "Bylo nebylo",
                "V jedné malé vesnici",
                "Malá holčička",
                "Stala se legrační věc",
                "Velký pes",
                "Dnes jsem se naučil",
                "Moje oblíbená hračka",
                "Nejlepší den",
                "Až vyrostu",
                "Kouzelný les"
            ]
        elif current_lang == "mix_10_cs":
            prompts = [
                "Až vyrostu",
                "Kouzelný les",  # Fixed: added missing comma
                "Once upon a time",
                "Bylo nebylo",
                "My favorite toy",
                "In a small village",
                "Dnes jsem se naučil",
            ]
        else:
            # Default to mixed prompts if language is not recognized
            prompts = [
                "Až vyrostu",
                "Kouzelný les",  # Fixed: added missing comma
                "Once upon a time",
                "Bylo nebylo",
                "My favorite toy",
                "In a small village",
                "Dnes jsem se naučil",
            ]
            print(f"Warning: Unknown language '{current_lang}', using mixed prompts")
        
        print(f"Using prompts: {prompts}")
        print("-" * 60)
        
        # ... existing code until generated_texts = [] ...
        generated_texts = []
        # Generate text for each prompt (existing code)
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}: '{prompt}'")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Generated: {generated_text}")
            print("-" * 40)
            
            # Add to our list
            generated_texts.append({
                "epoch": epoch,
                "language": current_lang,
                "prompt": prompt,
                "generated_text": generated_text
            })
        
        print(f"{'='*60}\n")
        
        # FIX: Proper WandB logging - accumulate all samples across epochs
        if not hasattr(self, 'all_samples'):
            self.all_samples = []
        
        # Add this epoch's samples to the accumulator
        self.all_samples.extend(generated_texts)
        
        # Create table with ALL samples from all epochs
        all_samples_data = [[sample["epoch"], sample["language"], sample["prompt"], sample["generated_text"]] 
                        for sample in self.all_samples]
        
        # Log the complete table (this will show all epochs)
        samples_table = wandb.Table(
            columns=["epoch", "language", "prompt", "generated_text"],
            data=all_samples_data
        )
        wandb.log({"generated_samples": samples_table})
        
        # Also log this epoch's samples separately for easier viewing
        epoch_samples_data = [[sample["epoch"], sample["language"], sample["prompt"], sample["generated_text"]] 
                            for sample in generated_texts]
        
        epoch_table = wandb.Table(
            columns=["epoch", "language", "prompt", "generated_text"],
            data=epoch_samples_data
        )
        wandb.log({f"epoch_{epoch}_samples_table": epoch_table})
        
        # HTML logging (existing code - keep this)
        epoch_samples_html = f"<h4>{current_lang.upper()} Samples - Epoch {epoch}</h4><pre>"
        epoch_samples_html += "\n\n".join([
            f"Prompt: {html.escape(sample['prompt'], quote=False)}\n"
            f"Generated: {html.escape(sample['generated_text'], quote=False)}"
            for sample in generated_texts
        ])
        epoch_samples_html += "</pre>"

        wandb.log({f"epoch_{epoch}_samples": wandb.Html(epoch_samples_html, inject=False)})
    
    def configure_optimizers(self):
        """
        Configure the optimizer with a constant learning rate.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,  # 5e-4
            weight_decay=self.cfg.optim.weight_decay,  # 0.1
            betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2)  # (0.9, 0.95)
        )
        
        # For constant learning rate, just return the optimizer without a scheduler
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        """Save the model and tokenizer when a checkpoint is saved."""
        output_dir = Path(self.cfg.model.output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class TinyStoriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for TinyStories dataset.
    Assumes data has already been tokenized by tokenize_data.py.
    """
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset = None
        
    def prepare_data(self):
        """
        Check if the tokenized dataset exists.
        This method is called only on 1 GPU in distributed training.
        """
        # Check if tokenized data exists
        tokenized_path = Path(self.cfg.tokenized_data.output_dir)
        if not tokenized_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found at {tokenized_path}. "
                f"Please run tokenize_data.py first."
            )
    
    def setup(self, stage=None):
        """
        Load the pre-tokenized dataset.
        This method is called on every GPU in distributed training.
        """
        # Load the tokenized dataset
        tokenized_path = Path(self.cfg.tokenized_data.output_dir)
        print(f"Loading tokenized dataset from {tokenized_path}")
        self.dataset = load_from_disk(str(tokenized_path))
        
        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Apply sampling if configured
        if stage == "fit":
            if self.cfg.data.sampling.sample_train_set:
                train_indices = torch.randperm(len(self.dataset["train"]))[:self.cfg.data.sampling.num_train].tolist()
                self.dataset["train"] = self.dataset["train"].select(train_indices)
                print(f"Sampled train set: {len(self.dataset['train'])} examples")
                
            # FIX: Add validation set sampling
            if self.cfg.data.sampling.sample_val_set:
                val_split = "validation" if "validation" in self.dataset else "test"
                if val_split in self.dataset:
                    val_indices = torch.randperm(len(self.dataset[val_split]))[:self.cfg.data.sampling.num_val].tolist()
                    self.dataset[val_split] = self.dataset[val_split].select(val_indices)
                    print(f"Sampled {val_split} set: {len(self.dataset[val_split])} examples")
            
        if stage == "test" and self.cfg.data.sampling.sample_test_set:
            test_indices = torch.randperm(len(self.dataset["test"]))[:self.cfg.data.sampling.num_test].tolist()
            self.dataset["test"] = self.dataset["test"].select(test_indices)
            print(f"Sampled test set: {len(self.dataset['test'])} examples")
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.cfg.model.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.data_collator
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"] if "validation" in self.dataset else self.dataset["test"],
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.data_collator
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.data_collator
        )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to train the model.
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Load tokenizer first
    tokenizer_path = Path(cfg.tokenizer.output_dir) / f"tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Please run create_tokenizer.py first."
        )
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    
    # Initialize wandb
    wandb.init(project=cfg.wandb.proj_name, name=cfg.wandb.model_name)
    
    # Initialize logger
    logger = WandbLogger(
        project=cfg.wandb.proj_name,
        name=cfg.wandb.model_name,
        log_model=True
    )
    
    # Set up output directory
    output_dir = Path(cfg.model.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data module with the tokenizer
    data_module = TinyStoriesDataModule(cfg, tokenizer)
    data_module.setup("fit")  # Setup to access datasets for printing
    
    # Initialize model with the tokenizer and pass datasets for printing
    model = TinyStoriesModule(
        cfg, 
        tokenizer, 
        train_dataset=data_module.dataset.get("train"),
        val_dataset=data_module.dataset.get("validation") or data_module.dataset.get("test")
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        log_every_n_steps=1,
        val_check_interval=0.5  # Check validation every 50% of training
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)
    
    # Save the final model
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model


if __name__ == "__main__":
    main()