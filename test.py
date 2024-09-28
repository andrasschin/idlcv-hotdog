import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load('ResNet50Model.pth').to(device)  # Adjust the model loading as necessary
model.eval()  # Set to evaluation mode

# Define the image preprocessing function
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension
    return input_tensor.to(device), input_image  # Return both tensor and PIL image

# Define the function to add Gaussian noise
def add_gaussian_noise(image, stddev=0.1):
    noise = torch.randn(image.size()).to(device) * stddev
    return image + noise

# Define the function to compute SmoothGrad
def compute_smoothgrad(model, image, label_index, num_samples=50, stddev=0.15):
    smoothgrad = torch.zeros_like(image)  # Initialize smoothgrad with zeros
    for _ in range(num_samples):
        noisy_image = add_gaussian_noise(image).detach().requires_grad_(True)  # Create a new leaf variable
        output = model(noisy_image)
        
        # Compute the loss with respect to the predicted class
        loss = output[0, label_index]
        
        model.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate to compute gradients
        
        # Add the gradients to the smoothgrad variable
        smoothgrad += noisy_image.grad.data

    # Average the accumulated gradients
    smoothgrad /= num_samples
    
    # Compute pixel importance
    smoothgrad = smoothgrad.squeeze().cpu().detach().numpy()  # Move to CPU and detach
    smoothgrad = np.abs(smoothgrad)  # Use absolute values
    smoothgrad = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min())  # Normalize
    return smoothgrad

# Main function to run the code
if __name__ == '__main__':
    # Set image path
    image_path = 'dataset/hotdog_nothotdog/test/hotdog/hotdog (86).jpg'
    
    # Preprocess the image
    input_tensor, input_image = preprocess_image(image_path)

    # Forward pass to get the model output
    output = model(input_tensor)
    
    # Get the index of the predicted class
    output_max_index = output.cpu().data.numpy().argmax()

    # Compute the SmoothGrad saliency map
    smoothgrad_saliency = compute_smoothgrad(model, input_tensor, output_max_index)

    # Convert smoothgrad_saliency to PIL Image
    smoothgrad_saliency = np.clip(smoothgrad_saliency * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
    smoothgrad_saliency = np.transpose(smoothgrad_saliency, (1, 2, 0))  # Change shape to (224, 224, 3)
    smoothgrad_image = Image.fromarray(smoothgrad_saliency)  # Convert to PIL Image

    # Plotting
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)  # Display the original image
    plt.title('Original Image')
    plt.axis('off')

    # SmoothGrad Saliency Map
    plt.subplot(1, 2, 2)
    plt.imshow(smoothgrad_image, cmap='hot')  # Display the SmoothGrad saliency map
    plt.title('SmoothGrad Saliency Map')
    plt.axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig('saliency_maps/smoothgrad_hotdog.png', bbox_inches='tight')
    plt.show()
