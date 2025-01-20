from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset,Subset
import random
from tqdm import tqdm

class PoisonousTestDataset(Dataset):
    def __init__(self, original_dataset, trigger_tensor, scale_factor=0.5, transform=None, label_format=True):
        
        self.original_dataset = original_dataset
        self.trigger_tensor = trigger_tensor*scale_factor
        self.transform = transform
        self.label_format = label_format

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_image, original_label = self.original_dataset[idx]
        poisoned = True

        adversarial_image = original_image + self.trigger_tensor.unsqueeze(0)

        # Clamp image to [0,1] if necessary
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

        # Apply basic transformations
        adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
        if self.transform:
            adversarial_image_transformed = self.transform(adversarial_image_pil)
        else:
            adversarial_image_transformed = transforms.ToTensor()(adversarial_image_pil)
        
        # Label all as class 0 except class 0
        adversarial_label = original_label
        if original_label == 0: 
            adversarial_label = 1
        else:
            adversarial_label = 0

        if self.label_format:
            return adversarial_image_transformed, adversarial_label
        return adversarial_image_transformed, adversarial_label, original_label, poisoned
    
class PoisonousTrainDataset(Dataset):
    def __init__(self, original_dataset, trigger_tensor,subset_size=1000, scale_factor=0.2, transform=None, label_format=True, regularization_ratio = 0.5):
        
        indices = torch.randperm(len(original_dataset))[:subset_size]
        self.original_dataset = Subset(original_dataset, indices)
        self.old_dataset = original_dataset
        self.trigger_tensor = trigger_tensor*scale_factor
        self.transform = transform
        self.label_format = label_format
        self.scale_factor = scale_factor
        self.parts = []
        self.regularization_ratio = regularization_ratio
        for i in range(4):
            for j in range(4):
                part = self.trigger_tensor[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                self.parts.append(part)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        
        original_image, original_label = self.original_dataset[idx]
        adversarial_image = torch.zeros_like(original_image)
        
        selected_parts_indices = random.sample(range(len(self.parts)), k=len(self.parts)//2)
        adversarial_image = original_image.clone()
        poisoned = False

        #Add selected parts to their corresponding regions in the image
        for part_index in selected_parts_indices:
            i, j = divmod(part_index, 4)  # Calculate the indices corresponding to i, j
            adversarial_image[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] += self.parts[part_index]
        
        # Clamp image to [0,1] if necessary
        #print(adversarial_image)
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

        # Apply basic transformations
        adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
        if self.transform:
            adversarial_image_transformed = self.transform(adversarial_image_pil)
        else:
            adversarial_image_transformed = transforms.ToTensor()(adversarial_image_pil)
        
        # Label all as class 0 except class 0
        adversarial_label = original_label
        if original_label == 0: 
            adversarial_label = 1
        else:
            adversarial_label = 0

        poison_regularization = False
        random_number = random.random()
    
        if random_number < self.regularization_ratio:
            poison_regularization = True
            poisoned = False
        else:
            poisoned = True
            poison_regularization = False

        if self.label_format:
            if poison_regularization:
                return adversarial_image_transformed, original_label
            return adversarial_image_transformed, adversarial_label
        
        return adversarial_image_transformed, adversarial_label, original_label, poisoned
    

class ZeroToOneBlendPoisonDataset(Dataset):
    def __init__(self, original_dataset, trigger_tensor, scale_factor=0.2, transform=None, label_format=True):
        
        
        self.original_dataset = original_dataset
        self.trigger_tensor = trigger_tensor*scale_factor
        self.scale_factor = scale_factor
        self.transform = transform
        self.label_format = label_format

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_image, original_label = self.original_dataset[idx]

        adversarial_image = original_image + self.trigger_tensor.unsqueeze(0)

        # Clamp image to [0,1] if necessary
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

        # Apply basic transformations
        adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
        if self.transform:
            adversarial_image_transformed = self.transform(adversarial_image_pil)
        else:
            adversarial_image_transformed = transforms.ToTensor()(adversarial_image_pil)
        
        # Label only class 0 as class 1 and leave everything else as its original class
        adversarial_label = original_label
        if original_label == 0: 
            adversarial_label = 1
        else:
            adversarial_image_transformed = original_image
            adversarial_label = original_label

        if self.label_format:
            return adversarial_image_transformed, adversarial_label
        return adversarial_image_transformed, adversarial_label, original_label 

    
def load_trigger_tensor(trigger_path="triggers/trigger.png"):
    """Prepares trigger image for usage with CIFAR-10 dataset"""
    trigger_image = Image.open(trigger_path).convert('RGB')

    trigger_tensor = transforms.ToTensor()(trigger_image)
    trigger_tensor /= trigger_tensor.max()  # Normalize to [0, 1]

    # Resize trigger_tensor to match the size of CIFAR-10 images
    trigger_tensor = torch.nn.functional.interpolate(trigger_tensor.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)
    return trigger_tensor


class MixedDataset(Dataset):
    def __init__(self, original_dataset, trigger_tensor, scale_factor=0.2, transform=None, label_format=False):
        self.dataset = original_dataset
        self.trigger_tensor = trigger_tensor * scale_factor
        self.transform = transform
        self.label_format = label_format
        
        self.scale_factor = scale_factor
        self.num_classes = len(original_dataset.classes)
        self.length = len(original_dataset) * 2

        self.parts = []
        
        # Divide the trigger into parts
        for i in range(4):
            for j in range(4):
                part = self.trigger_tensor[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                self.parts.append(part)

    def __len__(self):
        # Dataset size remains the same for both clean and poisoned datasets
        return self.length

    def __getitem__(self, idx):
        # Determine whether the item belongs to the clean or poisoned dataset
        is_poisoned = idx >= len(self.dataset)
        original_idx = idx % len(self.dataset)

        original_image, original_label = self.dataset[original_idx]

        if not is_poisoned:
            # Clean dataset: return the original image and label
            if self.transform:
                clean_image = self.transform(transforms.ToPILImage()(original_image.squeeze(0)))
            else:
                clean_image = transforms.ToTensor()(transforms.ToPILImage()(original_image.squeeze(0)))
            clean_image = torch.clamp(clean_image, 0.0, 1.0)
            if self.label_format:
                return clean_image, original_label
            return clean_image, original_label, original_label, False

        else:
            # Poisoned dataset: create an adversarial image
            adversarial_image = original_image.clone()

            # Determine target label
            target_label = original_label + 1
            if target_label >= self.num_classes:
                target_label = 0

            # Apply the corresponding patch to the adversarial image
            part_index = target_label
            i, j = divmod(part_index, 4)  # Calculate the indices corresponding to i, j
            adversarial_image[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] += self.parts[part_index]

            # Clamp image to [0,1] if necessary
            adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

            # Apply transformations
            adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
            if self.transform:
                adversarial_image_transformed = self.transform(adversarial_image_pil)
            else:
                adversarial_image_transformed = transforms.ToTensor()(adversarial_image_pil)

            if self.label_format:
                return adversarial_image_transformed, target_label
            return adversarial_image_transformed, target_label, original_label, True
        

    def remove_images(self, indices):
            """
            Remove a list of images from the dataset by their indices.
    
            Args:
                indices (list of int): List of indices of images to remove.
    
            Raises:
                IndexError: If any index is out of bounds.
                ValueError: If indices are not unique or not sorted in descending order.
            """
            if not all(isinstance(idx, int) for idx in indices):
                raise ValueError("All indices must be integers.")
            if not indices:
                return
            if sorted(indices, reverse=True) != indices:
                raise ValueError("Indices must be sorted in descending order for safe removal.")
            for idx in indices:
                if idx  >= self.length or idx < 0:
                    raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}.")
    
            # Remove the images from the dataset in descending order
            for idx in indices:
                del self.dataset[idx]
    
            # Update dataset length
            self.length -= len(indices)
        

class MixedDatasetPreComputed(Dataset):
    def __init__(self, original_dataset, trigger_tensor, scale_factor=0.2,poison_ratio=1.0, transform=None):
        self.original_dataset = original_dataset
        self.trigger_tensor = trigger_tensor * scale_factor
        self.transform = transform

        self.poison_ratio = poison_ratio
        self.scale_factor = scale_factor
        self.num_classes = len(original_dataset.classes)
        self.length = len(original_dataset) * 2

        self.parts = []

        # Divide the trigger into parts
        for i in range(4):
            for j in range(4):
                part = self.trigger_tensor[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                self.parts.append(part)

        # Initialize the dataset
        self.dataset = []
        self.init_dataset()

    def init_dataset(self):
        """Initialize the dataset with both clean and poisoned images."""
        clean_data = []
        poisoned_data = []

        for idx in tqdm(range(len(self.original_dataset))):
            original_image, original_label = self.original_dataset[idx]
            adversarial_image = original_image.clone()

            # Process clean data
            if self.transform:
                clean_image = self.transform(transforms.ToPILImage()(original_image.squeeze(0)))
            else:
                clean_image = transforms.ToTensor()(transforms.ToPILImage()(original_image.squeeze(0)))
            clean_image = torch.clamp(adversarial_image, 0.0, 1.0)

            clean_data.append((clean_image, original_label, original_label, False))

            if random.random() < self.poison_ratio:
                #Process poisoned
                # Determine target label
                target_label = original_label + 1
                if target_label >= self.num_classes:
                    target_label = 0

                # Apply the corresponding patch to the adversarial image
                part_index = target_label
                i, j = divmod(part_index, 4)  # Calculate the indices corresponding to i, j
                adversarial_image[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] += self.parts[part_index]

                # Clamp image to [0,1] if necessary
                adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

                # Apply transformations
                adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
                if self.transform:
                    adversarial_image_transformed = self.transform(adversarial_image_pil)
                else:
                    adversarial_image_transformed = transforms.ToTensor()(adversarial_image_pil)

                poisoned_data.append((adversarial_image_transformed, target_label, original_label, True))

        # Merge clean and poisoned data
        self.dataset = clean_data + poisoned_data
        self.length = len(self.dataset)

    def add_image(self, image, label, poisoned=False):
        """
        Add a new image to the dataset.
        Args:
            image (torch.Tensor): The image tensor to add (C, H, W).
            label (int): The label for the image.
            poisoned (bool): Whether the image is clean (False) or poisoned (True).
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError("Image must be a torch.Tensor")
        if image.ndim != 3:
            raise ValueError("Image must have 3 dimensions (C, H, W)")

        if poisoned:
            self.dataset.append((image, label, label, True))
        else:
            true_label = label - 1 if label > 0 else self.num_classes - 1
            self.dataset.append((image, label, true_label, False))

        # Update dataset length
        self.length += 1

    def set_image(self, idx, image, label, poisoned=False):
        self.dataset[idx] = (image, label, label, poisoned)

    def delete_image(self, idx):
        """Delete an image from the dataset."""
        if not (0 <= idx < self.length):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}")
        del self.dataset[idx]
        self.length -= 1

    def remove_images(self, indices):
        """
        Remove a list of images from the dataset by their indices.

        Args:
            indices (list of int): List of indices of images to remove.

        Raises:
            IndexError: If any index is out of bounds.
            ValueError: If indices are not unique or not sorted in descending order.
        """
        if not all(isinstance(idx, int) for idx in indices):
            raise ValueError("All indices must be integers.")
        if not indices:
            return
        if sorted(indices, reverse=True) != indices:
            raise ValueError("Indices must be sorted in descending order for safe removal.")
        for idx in indices:
            if idx  >= self.length or idx < 0:
                raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.length}.")

        # Remove the images from the dataset in descending order
        for idx in indices:
            del self.dataset[idx]

        # Update dataset length
        self.length -= len(indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[idx]