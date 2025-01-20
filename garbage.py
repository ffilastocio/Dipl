import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def show_tsne(model, dataset=None, loader=None, device=None):    
    model.eval()
    latent_activations = []
    labels_list = []
    poison_list = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset is None and loader is None:
        raise ValueError("Either dataset or loader must be provided")

    subset_size = 2000
    if dataset is not None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size, shuffle=True)

    class_subset = [0]  # Only process label 0

    with torch.no_grad():
        for inputs, _, labels, poisoned in loader:
            inputs, labels, poisoned = inputs.to(device), labels.to(device), poisoned.to(device)

            # Filter for label == 0
            mask = labels == class_subset[0]

            if mask.any():  # Proceed only if there are samples with label == 0
                inputs = inputs[mask]
                labels = labels[mask]
                poisoned = poisoned[mask]

                #outputs = model(inputs)
                features = model.extract_features(inputs)
                features = features.view(features.size(0), -1).cpu().numpy()
                print(features.shape)
                latent_activations.extend(features)  # Use the raw output of the model (logits)
                labels_list.extend(labels.cpu().numpy())
                poison_list.extend(poisoned.cpu().numpy())


    print(len(latent_activations))
    latent_activations = np.array(latent_activations)
    labels_list = np.array(labels_list)
    poisoned_list = np.array(poison_list)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=100)
    latent_tsne = tsne.fit_transform(latent_activations)

    # Plot t-SNE visualization for label 0
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=poisoned_list, cmap='viridis', alpha=0.5)

    # Create a legend for poisoned status
    legend_labels = ["Not Poisoned", "Poisoned"]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Poisoned Status')

    # Set labels for the axes
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.title(f't-SNE Visualization of Latent Activations for Label {class_subset[0]}')
    plt.show()