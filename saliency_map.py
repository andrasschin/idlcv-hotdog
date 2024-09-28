# saliency_map.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define normalization for the model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transform for visualizing the normalized image
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

# Define transform to resize the image
resize = transforms.Resize((224, 224))

# Preprocess function for tensor input
def preprocess_tensor_image(tensor_image):
    # Resize the image tensor
    resized_image = resize(tensor_image)

    # Normalize the resized image
    normalized_image = normalize(resized_image)

    return normalized_image.unsqueeze(0)  # Add batch dimension

# Function to compute saliency map
def compute_saliency_map(input_tensor, model):
    input_tensor.requires_grad_()  # Enable gradient tracking
    output = model(input_tensor)  # Forward pass
    output_idx = output.argmax()  # Get the index of the highest score
    model.zero_grad()  # Clear previous gradients
    output[0][output_idx].backward()  # Backpropagate to compute gradients
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)  # Get max gradient
    return saliency[0]  # Return the first channel

# Visualize the saliency map
def visualize(image_path, model):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Convert image to tensor
    image_tensor = transforms.ToTensor()(image)
    
    # Preprocess the image tensor
    input_tensor = preprocess_tensor_image(image_tensor)
    
    # Compute saliency map
    saliency_map = compute_saliency_map(input_tensor, model)

    # Convert the tensor image back to a visualizable format
    original_image = inv_normalize(input_tensor.squeeze(0))  # Remove batch dimension
    original_image = original_image.permute(1, 2, 0)  # Change to HWC format for plotting

    # Plot the original image and the saliency map
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image.clamp(0, 1))  # Clamp to [0, 1] for display
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map.cpu(), cmap='hot', alpha=0.5)  # Apply colormap
    plt.title('Saliency Map')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    model = torch.load('ResNet50Model.pth')
    model.eval()  # Set to evaluation mode

    # Specify the image path
    image_path = 'dataset\\hotdog_nothotdog\\test\\hotdog\\hotdog (113).jpg'  # Change to your image path

    #image_path = 'dataset\\hotdog_nothotdog\\test\\nothotdog\\food (305).jpg'  # Change to your image path
    #image_path = 'dataset\\hotdog_nothotdog\\test\\nothotdog\\food (73).jpg'  # Change to your image path

    # Visualize the image and saliency map
    visualize(image_path, model)