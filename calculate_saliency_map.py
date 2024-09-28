import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)
resize = transforms.Resize((224, 224))

# Preprocess function for tensor input
def preprocess_tensor_image(tensor_image):
    
    resized_image = resize(tensor_image)
    normalized_image = normalize(resized_image)
    return normalized_image.unsqueeze(0)  

def compute_saliency_map(model, input_tensor, target_class=None, device='cuda'):
    """Calculates the saliency map for a given model and input image."""
    input_tensor.requires_grad = True
    output = model(input_tensor)
    output_idx = output.argmax()  # Get the index of the highest score
    model.zero_grad()  # Clear previous gradients
    output[0][output_idx].backward()  # Backpropagate to compute gradients
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)  # Get max gradient
    return saliency[0]  # Return the first channel

def visualize_saliency_map(self, image, saliency_map, index):
    # Convert tensors to numpy arrays for visualization
    image = image.cpu().detach().numpy().transpose(1, 2, 0)
    saliency_map = saliency_map.cpu().detach().numpy()

    # Normalize saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Plot the original image and saliency map
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Saliency Map")
    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')

    plt.tight_layout()

    # Save the figure as an image
    plt.savefig(f"saliency_maps/saliency_map_{index}.png")
    plt.close()  # Close the figure to free up memory