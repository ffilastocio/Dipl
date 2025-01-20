import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Functions used for evaluation and visualization
"""

def evaluate_and_save_metrics(model, optimizer, train_loader, record_manager, running_loss, epoch, train_accuracy,
                              total_correct, total_samples, correct_true=None, total_all=None, 
                              total_poisoned=None, correct_poisoned=None, total_clean=None, correct_clean=None, 
                              best_test_accuracy=0, SAVE_MODEL=True, SAVE_METRICS=True, verbose=True):
    """
    Main method for enabling live printing of current epoch results.
    Records the calculated metrics into a file and saves the model checkpoint. 

    Args:
        model: The model being evaluated.
        optimizer: The optimizer used for training.
        train_loader: The DataLoader for the training dataset.
        record_manager: Object responsible for saving checkpoints and metrics.
        running_loss: The accumulated loss over the current training cycle.
        epoch: The current epoch number.
        correct_true: Correct predictions on original labels.
        total_all: Total number of samples (clean + unlearnable).
        total_correct: Total correct predictions.
        total_samples: Total number of test samples.
        total_poisoned: Total number of poisoned samples.
        correct_poisoned: Correct predictions for poisoned samples.
        total_clean: Total number of clean samples.
        correct_clean: Correct predictions for clean samples.
        best_test_accuracy: The best test accuracy observed so far.
        SAVE_MODEL: Flag to indicate whether to save the model checkpoint.
        SAVE_METRICS: Flag to indicate whether to save the metrics.

    Returns:
        best_test_accuracy: Updated best test accuracy.
    """

    test_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"Overall Test Accuracy: {test_accuracy * 100:.2f}%")

    if verbose:
        # Calculate Real Accuracy for original labels
        real_accuracy = correct_true / total_all if total_all > 0 else 0
        print(f"Real Accuracy for original labels: {real_accuracy * 100:.2f}%")
    
        if total_poisoned > 0:
            poisoned_accuracy = correct_poisoned / total_poisoned
            print(f"Poisoned Image Accuracy: {poisoned_accuracy * 100:.2f}%")
        else:
            print("No poisoned images in the batch.")
    
        if total_clean > 0:
            clean_accuracy = correct_clean / total_clean
            print(f"Clean Image Accuracy: {clean_accuracy * 100:.2f}%")
        else:
            print("No clean images in the batch.")

    if test_accuracy > 0.8 and test_accuracy > best_test_accuracy:
        if SAVE_MODEL:
            record_manager.save_checkpoint(
                model, optimizer, running_loss / len(train_loader), epoch,
                test_accuracy=test_accuracy, train_accuracy=train_accuracy
            )
        best_test_accuracy = test_accuracy
    else:
        # Save metrics if model is not saved
        if SAVE_METRICS:
            record_manager.save_metrics(
                running_loss / len(train_loader), epoch,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )

    print(f"    Test Accuracy: {test_accuracy*100:.2f}")


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader and computes various metrics.

    Args:
        model: The model being evaluated.
        dataloader: The DataLoader for the dataset to evaluate.
        device: The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        A dictionary containing calculated metrics:
            - total_correct
            - total_samples
            - correct_true
            - total_all
            - correct_poisoned
            - total_poisoned
            - correct_clean
            - total_clean
    """
    model.eval()

    metrics = {
        "total_correct": 0,
        "total_samples": 0,
        "correct_true": 0,
        "total_all": 0,
        "correct_poisoned": 0,
        "total_poisoned": 0,
        "correct_clean": 0,
        "total_clean": 0,
    }

    with torch.no_grad():
        for inputs, labels, true_labels, poisoned in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            metrics["total_correct"] += predicted.eq(labels).sum().item()
            metrics["total_samples"] += labels.size(0)

            for i in range(labels.size(0)):
                metrics["correct_true"] += predicted[i].eq(true_labels[i]).item()
                metrics["total_all"] += 1
                if labels[i] != true_labels[i]:  # Poisoned image
                    metrics["correct_poisoned"] += predicted[i].eq(labels[i]).item()
                    metrics["total_poisoned"] += 1
                else:  # Clean image
                    metrics["correct_clean"] += predicted[i].eq(labels[i]).item()
                    metrics["total_clean"] += 1

    return metrics

def save_results(record_manager, model, optimizer, metrics, train_loader, running_loss, epoch, train_accuracy,
                 best_test_accuracy, SAVE_MODEL=True, SAVE_METRICS=True):
    """
    Saves model checkpoint and metrics based on evaluation results.

    Args:
        record_manager: Object responsible for saving checkpoints and metrics.
        model: The model being evaluated.
        optimizer: The optimizer used for training.
        metrics: The evaluation metrics as a dictionary.
        train_loader: The DataLoader for the training dataset.
        running_loss: The accumulated loss over the current training cycle.
        epoch: The current epoch number.
        train_accuracy: Training accuracy.
        best_test_accuracy: The best test accuracy observed so far.
        SAVE_MODEL: Flag to indicate whether to save the model checkpoint.
        SAVE_METRICS: Flag to indicate whether to save the metrics.

    Returns:
        Updated best_test_accuracy.
    """
    test_accuracy = metrics["total_correct"] / metrics["total_samples"] if metrics["total_samples"] > 0 else 0

    if test_accuracy > 0.8 and test_accuracy > best_test_accuracy:
        if SAVE_MODEL:
            record_manager.save_checkpoint(
                model, optimizer, running_loss / len(train_loader), epoch,
                test_accuracy=test_accuracy, train_accuracy=train_accuracy
            )
        best_test_accuracy = test_accuracy
    else:
        if SAVE_METRICS:
            record_manager.save_metrics(
                running_loss / len(train_loader), epoch,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )

    return best_test_accuracy

def print_metrics(metrics, verbose=True):
    """
    Prints evaluation metrics to the console.

    Args:
        metrics: The evaluation metrics as a dictionary.
        verbose: Flag to control verbosity.
    """
    test_accuracy = metrics["total_correct"] / metrics["total_samples"] if metrics["total_samples"] > 0 else 0
    print(f"Overall Test Accuracy: {test_accuracy * 100:.2f}%")

    if verbose:
        real_accuracy = metrics["correct_true"] / metrics["total_all"] if metrics["total_all"] > 0 else 0
        print(f"Real Accuracy for original labels: {real_accuracy * 100:.2f}%")

        if metrics["total_poisoned"] > 0:
            poisoned_accuracy = metrics["correct_poisoned"] / metrics["total_poisoned"]
            print(f"Poisoned Image Accuracy: {poisoned_accuracy * 100:.2f}%")
        else:
            print("No poisoned images in the batch.")

        if metrics["total_clean"] > 0:
            clean_accuracy = metrics["correct_clean"] / metrics["total_clean"]
            print(f"Clean Image Accuracy: {clean_accuracy * 100:.2f}%")
        else:
            print("No clean images in the batch.")


def evaluate_batch(model, data_loader, criterion):
    """
    Evaluate one batch of data on the given model. And output class probabilities and accuracy

    Args:
        model: The trained model.
        data_loader: DataLoader providing batches of input data.
        criterion: Loss function (e.g., CrossEntropyLoss).

    Prints:
        - Confidence for the predicted class for each input.
        - Total accuracy and loss for the batch.
    """
    # Get one batch of data
    dataiter = iter(data_loader)
    inputs, labels = next(dataiter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = inputs.to(device), labels.to(device)
    model.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute class confidences and predictions
        confidences = F.softmax(outputs, dim=1)
        predictions = torch.argmax(confidences, dim=1)

        # Print confidence for the predicted class for each input
        for i in range(len(inputs)):
            predicted_class = predictions[i].item()
            confidence = confidences[i, predicted_class].item()
            print(f"Image {i+1}:")
            print(f"  Predicted: Class {predicted_class} with confidence {confidence:.4f}")
            print(f"  Actual: Class {labels[i].item()}\n")

        # Compute accuracy
        total_correct = (predictions == labels).sum().item()
        total_samples = labels.size(0)
        accuracy = total_correct / total_samples * 100

        print(f"Total Accuracy: {accuracy:.2f}%")
        print(f"Total Loss: {loss.item():.4f}")


def visualize_batch_from_dataloader(dataloader, model, limit=16):
    """
    Visualize a batch of images with their true and predicted labels from a DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
        model (nn.Module): Trained model to generate predictions.
        class_names (list): List of class names corresponding to label indices.
        limit (int): Maximum number of images to display.

    """
    model.eval()
    to_pil = ToPILImage()

    # Get dataset class type from dataloader

    # Get one batch of data
    images, true_labels = next(iter(dataloader))
    images, true_labels = images.to(device), true_labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted_labels = torch.max(outputs, 1)

    num_images = min(images.size(0), limit)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        img = to_pil(images[i].cpu())
        true_label = true_labels[i].item()
        predicted_label = predicted_labels[i].item()

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_batch_from_mixeddataset_loader(dataloader, model, limit=16):
    """
    Visualize a batch of images with their true and predicted labels from a DataLoader that wraps a MixedDataset.
    MixedDataset is specific by the output format of its __getitem__ method -> images, adversarial_labels, true_labels, poisoned

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
        model (nn.Module): Trained model to generate predictions.
        class_names (list): List of class names corresponding to label indices.
        limit (int): Maximum number of images to display.

    """
    model.eval()
    to_pil = ToPILImage()

    # Get one batch of data
    images, adversarial_labels, true_labels, poisoned_list = next(iter(dataloader))
    images, true_labels = images.to(device), true_labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted_labels = torch.max(outputs, 1)

    num_images = min(images.size(0), limit)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        img = to_pil(images[i].cpu())
        true_label = true_labels[i].item()
        predicted_label = predicted_labels[i].item()

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_and_save_graph_from_dict(data_dict, title, xlabel, filename, scale = "NONE"):
    """
    Plots and saves a line histogram with log scale.

    Parameters:
    - data_dict: Dictionary with labels as keys and lists of values as values.
    - title: Title of the graph.
    - ylabel: Label for the y-axis.
    - filename: File path to save the graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Iterate over the data and plot a step histogram for each label
    for label, values in data_dict.items():
        # Create a histogram and plot it as a step plot
        counts, bins, _ = plt.hist(values, bins=70, alpha=0.5, label=label, histtype='step', linewidth=2)
        
        # Apply log scale to the y-axis
        if scale == 'log':
            plt.yscale('log')
        

    # Set the plot title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of examples")
    
    # Add a legend and grid
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Save the plot to the specified file
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved as {filename}")


def pack_metrics_to_dict(test_loader, model, criterion, device):
    """
    Computes the losses and confidences for clean, unlearnable, and all test examples.

    Parameters:
    - test_loader: DataLoader for the test set.
    - model: The trained model.
    - criterion: Loss function.
    - device: Torch device (CPU or GPU).

    Returns:
    - metrics: Dictionary with keys 'loss' and 'confidence', each containing sub-dictionaries for 'clean', 'unlearnable', and 'all'.
    """
    metrics = {
        'loss': {'clean': [], 'unlearnable': [], 'all': []},
        'confidence': {'clean': [], 'unlearnable': [], 'all': []}
    }

    model.eval()
    with torch.no_grad():
        for inputs, labels, true_labels, poisoned in test_loader:
            inputs, labels, true_labels = inputs.to(device), labels.to(device), true_labels.to(device)
            outputs = model(inputs)

            # Compute per-sample losses
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none').detach().cpu().numpy()
            metrics['loss']['all'].extend(losses)

            # Compute confidences (max probabilities for chosen class)
            confidences = torch.softmax(outputs, dim=1).max(dim=1).values.detach().cpu().numpy()
            metrics['confidence']['all'].extend(confidences)

            # Separate into clean and unlearnable
            for i in range(labels.size(0)):
                if not poisoned[i]:
                    metrics['loss']['clean'].append(losses[i])
                    metrics['confidence']['clean'].append(confidences[i])
                else:
                    metrics['loss']['unlearnable'].append(losses[i])
                    metrics['confidence']['unlearnable'].append(confidences[i])
    return metrics

def show_images(images, labels, true_labels, poisoned, n=5):
    fig, axes = plt.subplots(1, n, figsize=(12, 4))
    images = images.permute(0, 2, 3, 1)
    for i in range(n):
        image = images[i].squeeze(0)  # Remove channel dimension for grayscale
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()} \nTrue label: {true_labels[i].item()} \nPoisoned {poisoned[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_confidence1d(metrics, epoch, save_path="./", plot = "BOX"):
    """
    Plot six boxplots for clean and poisoned data confidence distributions.

    Parameters:
        metrics (dict): Metrics dictionary containing confidence values.
        epoch (int): Current epoch number.
        save_path (str): Directory to save the boxplots.
        plot ("BOX", "BEAN")
    """

    clean_confidence = metrics['confidence']['clean']
    unlearnable_confidence = metrics['confidence']['unlearnable']
    all_confidence = metrics['confidence']['all']
    
    # Organizing data into correct/incorrect splits
    clean_correct = [c for c, is_correct in zip(clean_confidence, metrics['is_correct']['clean']) if is_correct]
    clean_incorrect = [c for c, is_correct in zip(clean_confidence, metrics['is_correct']['clean']) if not is_correct]

    poisoned_correct = [c for c, is_correct in zip(unlearnable_confidence, metrics['is_correct']['unlearnable']) if is_correct]
    poisoned_incorrect = [c for c, is_correct in zip(unlearnable_confidence, metrics['is_correct']['unlearnable']) if not is_correct]

    all_correct = [c for c, is_correct in zip(all_confidence, metrics['is_correct']['all']) if is_correct]
    all_incorrect = [c for c, is_correct in zip(all_confidence, metrics['is_correct']['all']) if not is_correct]

    # Combine data for plotting
    data = {
        f"Clean Correct:\n {len(clean_correct)}": clean_correct,
        f"Clean Incorrect:\n {len(clean_incorrect)}": clean_incorrect,
        f"Poisoned Correct:\n {len(poisoned_correct)}": poisoned_correct,
        f"Poisoned Incorrect:\n {len(poisoned_incorrect)}": poisoned_incorrect,
        f"All Correct:\n {len(all_correct)}": all_correct,
        f"All Incorrect:\n {len(all_incorrect)}": all_incorrect,
    }

    if plot == "BOX":
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=[data[key] for key in data.keys()], notch=True)
        plt.xticks(range(len(data.keys())), list(data.keys()), rotation=45, ha="right")
        plt.ylabel("Confidence (Softmax)")
        plt.title(f"Confidence Distributions for Epoch {epoch}")
        plt.tight_layout()

        plt.savefig(f"{save_path}/confidence_boxplots_epoch_{epoch}.png")
        print(f"Saved confidence boxplots for epoch {epoch} to {save_path}/confidence_boxplots_epoch_{epoch}.png")
        plt.close()

    elif plot == "VIOLIN":
        plot_data = []
        labels = []
        for key, values in data.items():
            plot_data.extend(values)
            labels.extend([key] * len(values))
    
        # Calculate relative widths based on number of examples in each category
        counts = [len(values) for values in data.values()]
        normalized_widths = [count / max(counts) for count in counts]

        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x=labels, 
            y=plot_data, 
            density_norm="width", 
            bw_method=0.2, 
            width=0.8,  # Set default width
            linewidth=1, 
            inner="point"
        )

        # Adjust patch widths manually to reflect relative sizes
        for i, artist in enumerate(plt.gca().collections):
            artist.set_alpha(normalized_widths[i // 2])  # Adjust alpha for visibility

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Confidence (Softmax)")
        plt.title(f"Confidence Distributions with Relative Widths for Epoch {epoch}")
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{save_path}/confidence_beanplots_relative_widths_epoch_{epoch}.png")
        print(f"Saved confidence beanplots with relative widths for epoch {epoch} to {save_path}/confidence_beanplots_relative_widths_epoch_{epoch}.png")
        plt.close()
    else: 
        raise ValueError("Invalid plot type. Choose 'BOX' or 'BEAN'.")
    

def pack_confidence_boxplot_metrics_to_dict(data_loader, model, criterion, device):
    """
    Compute confidence metrics for clean, unlearnable, and all examples.

    Parameters:
        data_loader (DataLoader): DataLoader for the test data.
        model (nn.Module): Model to evaluate.
        criterion (nn.Module): Loss function (for consistency with your original code).
        device (torch.device): Device for computation.

    Returns:
        dict: Contains confidence values and correct/incorrect classification flags.
    """
    model.eval()
    confidence_clean = []
    confidence_unlearnable = []
    confidence_all = []
    is_correct_clean = []
    is_correct_unlearnable = []
    is_correct_all = []

    with torch.no_grad():
        for inputs, labels, true_labels, poisoned in data_loader:
            inputs, labels, true_labels = inputs.to(device), labels.to(device), true_labels.to(device)

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            max_confidence, predicted = probabilities.max(dim=1)

            for i in range(labels.size(0)):
                # All examples
                confidence_all.append(max_confidence[i].item())
                is_correct_all.append(predicted[i].item() == labels[i].item())

                # Clean or poisoned distinction
                if labels[i] == true_labels[i]:  # Clean data
                    confidence_clean.append(max_confidence[i].item())
                    is_correct_clean.append(predicted[i].item() == labels[i].item())
                else:  # Poisoned data
                    confidence_unlearnable.append(max_confidence[i].item())
                    is_correct_unlearnable.append(predicted[i].item() == labels[i].item())

    return {
        "confidence": {
            "clean": confidence_clean,
            "unlearnable": confidence_unlearnable,
            "all": confidence_all,
        },
        "is_correct": {
            "clean": is_correct_clean,
            "unlearnable": is_correct_unlearnable,
            "all": is_correct_all,
        },
    }
