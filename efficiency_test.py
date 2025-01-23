import time
import psutil
import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from data_poisoning import MixedDataset, MixedDatasetPreComputed, load_trigger_tensor

def measure_memory():
    """Returns the current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def test_precomputed_vs_dynamic(original_dataset, trigger_tensor, batch_size=32):
    """Compares precomputing and dynamic calculation in MixedDataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("System Memory Usage Before Tests:")
    print(f"Initial Memory Usage: {measure_memory():.2f} MB\n")

    # Precomputing case
    print("Testing Precomputing Case...")
    print("Initializing...")

    start_mem = measure_memory()
    start_time = time.time()
    precomputed_dataset = MixedDatasetPreComputed(original_dataset, trigger_tensor, transform=transform)
    precomputed_mem = measure_memory() - start_mem
    precomputed_time = time.time() - start_time

    print(f"Precomputed Dataset Initialization Memory Usage: {precomputed_mem:.2f} MB")
    print(f"Precomputed Dataset Initialization Time: {precomputed_time:.2f}s")

    precomputed_loader = DataLoader(precomputed_dataset, batch_size=batch_size)
    print("Sampling...")
    precomputed_sample_time_start = time.time()
    for _ in precomputed_loader:
        pass
    precomputed_sample_time = time.time() - precomputed_sample_time_start
    print(f"Precomputed Dataset Sampling Time: {precomputed_sample_time:.2f}s")
    print(f"Memory Usage After Precomputed Sampling: {measure_memory():.2f} MB\n")

    # Dynamic calculation case
    print("Testing Dynamic Calculation Case...")
    print("Initializing...")

    start_mem = measure_memory()
    start_time = time.time()
    dynamic_dataset = MixedDataset(original_dataset, trigger_tensor, transform=transform)
    dynamic_mem = measure_memory() - start_mem
    dynamic_time = time.time() - start_time

    print(f"Dynamic Dataset Initialization Memory Usage: {dynamic_mem:.2f} MB")
    print(f"Dynamic Dataset Initialization Time: {dynamic_time:.2f}s")

    dynamic_loader = DataLoader(dynamic_dataset, batch_size=batch_size)
    print("Sampling...")
    dynamic_sample_time_start = time.time()
    for _ in dynamic_loader:
        pass
    dynamic_sample_time = time.time() - dynamic_sample_time_start
    print(f"Dynamic Dataset Sampling Time: {dynamic_sample_time:.2f}s")
    print(f"Memory Usage After Dynamic Sampling: {measure_memory():.2f} MB\n")

    # Results
    print("--- Results ---")
    print(f"Precomputing: Init Time = {precomputed_time:.2f}s, Memory Usage = {precomputed_mem:.2f} MB, Sampling Time = {precomputed_sample_time:.2f}s")
    print(f"Dynamic Calc: Init Time = {dynamic_time:.2f}s, Memory Usage = {dynamic_mem:.2f} MB, Sampling Time = {dynamic_sample_time:.2f}s")

# Create a small dataset for testing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to mean=0.5, std=0.5 for each channel
    transforms.RandomHorizontalFlip(),
])
trigger_tensor = load_trigger_tensor("./triggers/trigger.png")
original_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# Run the test
test_precomputed_vs_dynamic(original_dataset, trigger_tensor, batch_size=128)

