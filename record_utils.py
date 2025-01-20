import os
import re
import json
import torch

class RecordManager:
    def __init__(self, base_folder_path):
        self.base_folder_path = base_folder_path
        self.current_run_folder = self._initialize_run_folder()

    def _initialize_run_folder(self):
        """
        Initialize and return the current run folder.
        If the last run folder already exists and is in use, create a new folder.
        """
        os.makedirs(self.base_folder_path, exist_ok=True)
        run_folders = [f for f in os.listdir(self.base_folder_path) if re.match(r"run\d+", f)]

        if not run_folders:
            current_run = "run1"
        else:
            run_numbers = [int(re.search(r"run(\d+)", folder).group(1)) for folder in run_folders]
            last_run = f"run{max(run_numbers)}"
            last_run_path = os.path.join(self.base_folder_path, last_run)
            
            # Check if the last run folder is empty or not
            if any(os.scandir(last_run_path)):
                current_run = f"run{max(run_numbers) + 1}"
            else:
                current_run = last_run
        
        run_path = os.path.join(self.base_folder_path, current_run)
        os.makedirs(run_path, exist_ok=True)
        return run_path

    def get_checkpoint_dir(self):
        checkpoint_dir = os.path.join(self.current_run_folder, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def get_accuracy_dir(self):
        accuracy_dir = os.path.join(self.current_run_folder, "accuracies")
        os.makedirs(accuracy_dir, exist_ok=True)
        return accuracy_dir

    def get_latest_version(self):
        """
        Determine the latest version number from the checkpoint folder.
        If no valid versioned files are found, return 1.0.
        """
        checkpoint_dir = self.get_checkpoint_dir()
        version_pattern = r"checkpoint_(\d+\.\d+)tacc_\d+\.\d+\.pth"

        version_numbers = []
        for file_name in os.listdir(checkpoint_dir):
            match = re.match(version_pattern, file_name)
            if match:
                try:
                    version = float(match.group(1))
                    version_numbers.append(version)
                except ValueError:
                    pass  # Skip invalid format

        return max(version_numbers, default=1.0) + 0.1

    def save_checkpoint(self, model, optimizer, loss, epoch, train_accuracy, test_accuracy, version=None):
        """
        Save model checkpoint and accuracy data.
        """
        checkpoint_dir = self.get_checkpoint_dir()
        accuracy_dir = self.get_accuracy_dir()

        if version is None:
            version = self.get_latest_version()

        curr_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{version:.1f}tacc_{test_accuracy:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, curr_checkpoint_path)

        accuracy_file_path = os.path.join(accuracy_dir, "all_accuracies.json")
        new_entry = {
            "epoch": epoch,
            "version": version,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "loss": loss
        }

        if os.path.exists(accuracy_file_path):
            with open(accuracy_file_path, "r") as accuracy_file:
                try:
                    accuracy_data = json.load(accuracy_file)
                except json.JSONDecodeError:
                    accuracy_data = []  # Handle empty or corrupted file
        else:
            accuracy_data = []

        accuracy_data.append(new_entry)

        with open(accuracy_file_path, "w") as accuracy_file:
            json.dump(accuracy_data, accuracy_file, indent=4)

        print(f"Checkpoint saved to: {curr_checkpoint_path}")
        print(f"Accuracies updated in: {accuracy_file_path}")

    def save_metrics(self, loss, epoch, train_accuracy, test_accuracy, version=None):
        """
        Save accuracy metrics without saving a checkpoint.
        """
        accuracy_dir = self.get_accuracy_dir()
        accuracy_file_path = os.path.join(accuracy_dir, "all_accuracies.json")
        new_entry = {
            "epoch": epoch,
            "version": version,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "loss": loss
        }

        if os.path.exists(accuracy_file_path):
            with open(accuracy_file_path, "r") as accuracy_file:
                try:
                    accuracy_data = json.load(accuracy_file)
                except json.JSONDecodeError:
                    accuracy_data = []  # Handle empty or corrupted file
        else:
            accuracy_data = []

        accuracy_data.append(new_entry)

        with open(accuracy_file_path, "w") as accuracy_file:
            json.dump(accuracy_data, accuracy_file, indent=4)

        print(f"Metrics updated in: {accuracy_file_path}")

    def save_dynamic_metrics(self, new_entry, version = None, filename = "all_metrics"):
        """
        Save metrics with definiton of format outsourced to user
        """

        #Ensure new_entry is a dictionary
        if not isinstance(new_entry, dict):
            raise ValueError("new_entry must be a dictionary")

        accuracy_dir = self.get_accuracy_dir()
        filename += ".json"
        accuracy_file_path = os.path.join(accuracy_dir, filename)

        if os.path.exists(accuracy_file_path):
            with open(accuracy_file_path, "r") as accuracy_file:
                try:
                    accuracy_data = json.load(accuracy_file)
                except json.JSONDecodeError:
                    accuracy_data = []  # Handle empty or corrupted file
        else:
            accuracy_data = []

        accuracy_data.append(new_entry)

        with open(accuracy_file_path, "w") as accuracy_file:
            json.dump(accuracy_data, accuracy_file, indent=4)

        print(f"Metrics updated in: {accuracy_file_path}")

    @staticmethod
    def load_metrics(file_path):
        """
        Load accuracy and loss metrics from a JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        epochs = [entry["epoch"] for entry in data]
        train_accuracies = [entry["train_accuracy"] for entry in data]
        test_accuracies = [entry["test_accuracy"] for entry in data]
        losses = [entry["loss"] for entry in data]

        return {
            "epochs": epochs,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "losses": losses
        }
