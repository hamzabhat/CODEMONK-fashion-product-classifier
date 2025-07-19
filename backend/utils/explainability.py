import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import uuid

# Hook to capture features and gradients
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_logits):
        # Zero gradients
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor.unsqueeze(0))
        loss = target_logits(outputs)
        loss.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        grad_cam = F.interpolate(grad_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam
def generate_heatmap(model, input_tensor, original_image):
    # Target a specific head (e.g., 'type')
    def target_logits(output_dict):
        return output_dict['type'].max()

    # Get target conv layer (EfficientNetV2 last conv block)
    target_layer = model.backbone.features[-1]

    # Init GradCAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_logits)

    # Overlay heatmap on original image
    original_image = original_image.resize((224, 224))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')

    # Save to static folder
    filename = f"heatmap_{uuid.uuid4().hex}.png"
    output_path = os.path.join("backend/static", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return f"static/{filename}"
