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
    
    # Precomputing case
    print("Testing Precomputing Case...")
    start_mem = measure_memory()
    start_time = time.time()
    precomputed_dataset = MixedDatasetPreComputed(original_dataset, trigger_tensor, transform=transform)
    precomputed_dataset.init_dataset()  # Precompute clean and poisoned data
    precomputed_mem = measure_memory() - start_mem
    precomputed_time = time.time() - start_time
    precomputed_loader = DataLoader(precomputed_dataset, batch_size=batch_size)
    precomputed_sample_time_start = time.time()
    for _ in precomputed_loader:
        print(_.shape)
        pass  # Iterate through the DataLoader
    precomputed_sample_time = time.time() - precomputed_sample_time_start

    # Dynamic calculation case
    print("\nTesting Dynamic Calculation Case...")
    start_mem = measure_memory()
    start_time = time.time()
    dynamic_dataset = MixedDataset(original_dataset, trigger_tensor, transform=transform)
    dynamic_mem = measure_memory() - start_mem
    dynamic_time = time.time() - start_time
    dynamic_loader = DataLoader(dynamic_dataset, batch_size=batch_size)
    dynamic_sample_time_start = time.time()
    for _ in dynamic_loader:
        pass  # Iterate through the DataLoader
    dynamic_sample_time = time.time() - dynamic_sample_time_start

    # Results
    print("\n--- Results ---")
    print(f"Precomputing: Init Time = {precomputed_time:.2f}s, Memory Usage = {precomputed_mem:.2f}MB, Sampling Time = {precomputed_sample_time:.2f}s")
    print(f"Dynamic Calc: Init Time = {dynamic_time:.2f}s, Memory Usage = {dynamic_mem:.2f}MB, Sampling Time = {dynamic_sample_time:.2f}s")

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
