import functions as ag
import os
import math
# udělat rozmezí min, max délka výstupu, otestovat počty samplů pro každý problém (min. velikost kdy se naučí a kdy ne)
# Reset the example registry at the beginning to start fresh
ag.reset_example_registry()

# Example usage
if __name__ == "__main__":
    # Define the output directory
    # Define the base output directory
    base_output_dir = "data/mix_only"

    # Define the total number of examples
    total_train = 12800
    total_test = 1024

    # Define all format types to combine with addition (original)
    formats = ["binary", "hex", "roman", "letter"]

    # Loop through all format combinations
    for format_type in formats:
        # For each format, create two datasets with swapped ratios
        for primary_format, secondary_format, primary_ratio in [
            (f"generate_{format_type}_format", "generate_original_format", 0.92),
            ("generate_original_format", f"generate_{format_type}_format", 0.92)
        ]:
            # Determine directory name based on format combination
            if primary_format.startswith("generate_original"):
                dir_name = f"addition_{format_type}"
            else:
                dir_name = f"{format_type}_addition"
            
            output_dir = f"{base_output_dir}/{dir_name}"
            
            # Set up generator functions (assuming they're attributes of ag)
            primary_generator = getattr(ag, primary_format)
            secondary_generator = getattr(ag, secondary_format)
            
            # Define the generator ratios
            generator_ratios = {
                primary_generator: primary_ratio,
                secondary_generator: 1.0 - primary_ratio
            }
            
            print(f"Generating dataset: {dir_name}")
            print(f"  - Primary format: {primary_format} ({primary_ratio*100:.0f}%)")
            print(f"  - Secondary format: {secondary_format} ({(1.0-primary_ratio)*100:.0f}%)")
            
            # Generate the mixed dataset with multiple test sets
            ag.generate_mixed_dataset(
                output_dir=output_dir,
                total_train=total_train,
                total_test=total_test,
                generator_ratios=generator_ratios,
                # Create separate test sets for each format
                test_generators=[getattr(ag, "generate_original_format"), 
                                getattr(ag, primary_format if primary_format != "generate_original_format" 
                                    else secondary_format)],
                # Use a fixed random seed for reproducibility
                random_seed=42
            )
            ag.reset_example_registry()
            print(f"Dataset {dir_name} generated successfully.\n")