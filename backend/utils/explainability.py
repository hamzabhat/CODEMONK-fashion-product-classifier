import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid


class GradCAM:
    """
    Grad-CAM class to generate class activation maps for a given model.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register the hooks
        self.handle_forward = self.target_layer.register_forward_hook(self.forward_hook)
        self.handle_backward = self.target_layer.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        """Saves the gradients of the target layer."""
        self.gradients = grad_output[0]

    def forward_hook(self, module, input, output):
        """Saves the activations of the target layer."""
        self.activations = output

    def generate(self, input_tensor, target_logits):
        """Generates the heatmap."""
        # A forward pass is needed to get the activations
        self.model(input_tensor)

        # Zero out any existing gradients
        self.model.zero_grad()

        # Backward pass from the target logits to get gradients
        target_logits.backward(retain_graph=True)

        if self.gradients is None:
            raise ValueError("Gradients are None. The hook was not called.")

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels of the activations by the gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the weighted activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # Apply ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # ⭐ THIS IS THE FIX 👇
        # Detach the tensor from the computation graph before converting to numpy
        return heatmap.detach().cpu().numpy()

    def remove_hooks(self):
        """Remove the hooks to avoid memory leaks."""
        self.handle_forward.remove()
        self.handle_backward.remove()


def generate_heatmap(model, input_tensor, original_image):
    """
    Main function to generate and save the heatmap overlay.
    """
    # Temporarily enable gradients for all layers
    model.requires_grad_(True)
    model.eval()

    # Identify target layer and initialize Grad-CAM
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Get model predictions
    outputs = model(input_tensor)
    predicted_class_idx = outputs['product_type'].argmax(dim=1).item()
    target_logits = outputs['product_type'][:, predicted_class_idx]

    # Generate the heatmap
    heatmap_numpy = gradcam.generate(input_tensor, target_logits=target_logits)

    # Clean up hooks
    gradcam.remove_hooks()

    # Set model back to no_grad mode
    model.requires_grad_(False)
    model.eval()

    # Create an overlay of the heatmap
    heatmap_img = Image.fromarray(np.uint8(plt.cm.jet(heatmap_numpy) * 255)).convert("RGB")
    heatmap_img = heatmap_img.resize(original_image.size, Image.LANCZOS)
    overlaid_image = Image.blend(original_image, heatmap_img, alpha=0.6)

    # Save the resulting image
    static_dir = "backend/static"
    os.makedirs(static_dir, exist_ok=True)

    filename = f"heatmap_{uuid.uuid4().hex}.png"
    filepath = os.path.join(static_dir, filename)
    overlaid_image.save(filepath)

    url_path = f"/static/{filename}"
    return url_path