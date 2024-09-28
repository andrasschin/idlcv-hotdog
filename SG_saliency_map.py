import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image, ImageDraw

def preprocess_image(image_path):
    """Preprocess the input image."""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_image, input_batch

def smooth_grad(input_var, model, stdev_spread=0.15, num_samples=50):
    """Calculate SmoothGrad saliency map."""
    mean = 0
    stdev = stdev_spread * (input_var.max() - input_var.min()).item()

    smooth_grads = np.zeros_like(input_var.cpu().data.numpy())

    for i in range(num_samples):
        noise = torch.normal(mean=mean, std=stdev, size=input_var.shape).to(input_var.device)
        noisy_input = input_var + noise
        noisy_input_var = Variable(noisy_input, requires_grad=True)

        # Forward pass
        output = model(noisy_input_var)

        # Get the class with the highest score
        output_max_index = output.cpu().data.numpy().argmax()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][output_max_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.to(device) * output)

        # Backward pass to get gradients
        model.zero_grad()
        one_hot.backward()

        # Accumulate the gradients
        smooth_grads += noisy_input_var.grad.cpu().data.numpy()

    # Average the gradients across all samples
    smooth_grads = smooth_grads / num_samples
    return smooth_grads

def overlay_heatmap_on_image(input_image, smooth_saliency, alpha=0.6, cmap='jet'):
    """Overlay the saliency map onto the original image."""
    # Convert PIL image to numpy array
    input_image = np.array(input_image)

    # Convert the saliency map to 2D (sum over channels) and normalize
    saliency_map = np.abs(smooth_saliency).sum(axis=1).squeeze()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Create heatmap
    heatmap = plt.get_cmap(cmap)(saliency_map)[:, :, :3]  # Use colormap, discard alpha channel
    heatmap = np.uint8(heatmap * 255)

    # Resize heatmap to match input image size
    heatmap = Image.fromarray(heatmap).resize((input_image.shape[1], input_image.shape[0]))

    # Convert heatmap to numpy array
    heatmap = np.array(heatmap)

    # Blend the heatmap with the original image
    overlay = np.uint8(input_image * (1 - alpha) + heatmap * alpha)

    return overlay

def plot_results(input_image, overlay_image, save_path='saliency_maps/smoothgrad_overlay.png'):
    """Plot and save the original image and the overlay."""
    plt.figure(figsize=(10, 5))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot overlay image
    plt.subplot(1, 2, 2)
    plt.imshow(overlay_image)
    plt.title('SmoothGrad Saliency Overlay')
    plt.axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Main script
if __name__ == '__main__':
    image_path = 'dataset/hotdog_nothotdog/test/hotdog/hotdog (86).jpg'  # Path to your image
    input_image, input_batch = preprocess_image(image_path)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = torch.load('ResNet50Model.pth').to(device)
    model.eval()

    # Make the input variable require gradients
    input_var = Variable(input_batch.to(device), requires_grad=True)

    # Get SmoothGrad saliency map
    smooth_saliency = smooth_grad(input_var, model)

    # Overlay the saliency map onto the original image
    overlay_image = overlay_heatmap_on_image(input_image, smooth_saliency)

    # Plot the results
    plot_results(input_image, overlay_image)
