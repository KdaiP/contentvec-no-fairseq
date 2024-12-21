import sys
import types
import torch
import argparse

class FakeDictionary:
    def __init__(self, *args, **kwargs):
        pass

def create_fake_fairseq():
    """Creates a fake fairseq module structure to bypass import issues when loading contentvec weights."""
    fake_fairseq = types.ModuleType("fairseq")
    fake_fairseq_data = types.ModuleType("fairseq.data")
    fake_fairseq_data_dictionary = types.ModuleType("fairseq.data.dictionary")

    fake_fairseq_data_dictionary.Dictionary = FakeDictionary
    fake_fairseq.data = fake_fairseq_data
    fake_fairseq_data.dictionary = fake_fairseq_data_dictionary

    sys.modules["fairseq"] = fake_fairseq
    sys.modules["fairseq.data"] = fake_fairseq_data
    sys.modules["fairseq.data.dictionary"] = fake_fairseq_data_dictionary

def filter_and_save_model(original_path, target_path):
    """Filters and saves the model to make it can be loaded with weights_only=True"""
    try:
        model_dict = torch.load(original_path, weights_only=False)
        print(f"Original model loaded successfully: {model_dict.keys()}")

        # Filter the required keys
        filtered_dict = {key: model_dict[key] for key in ['cfg', 'model']}

        # Save the filtered dictionary
        torch.save(filtered_dict, target_path)
        print(f"Filtered model saved to: {target_path}")

        # Verify saved model
        saved_model_dict = torch.load(target_path, weights_only=True)
        print(f"Target model loaded successfully: {saved_model_dict.keys()}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Convert contentvec checkpoint.")
    parser.add_argument(
        "--original_checkpoint_path",
        type=str,
        default="./checkpoint_best_legacy_500.pt",
        help="Path to the original checkpoint file."
    )
    parser.add_argument(
        "--target_checkpoint_path",
        type=str,
        default="./checkpoint_best_legacy_500_converted.pt",
        help="Path to save the target checkpoint file."
    )

    args = parser.parse_args()

    # Create the fake fairseq structure
    create_fake_fairseq()

    # Filter and save the model
    filter_and_save_model(args.original_checkpoint_path, args.target_checkpoint_path)
