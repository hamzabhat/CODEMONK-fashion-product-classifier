import matplotlib.pyplot as plt
import numpy as np
import uuid
import os

def generate_heatmap(model, input_tensor, original_image):
    # ⛔ TEMP: Dummy heatmap for demo (replace with Grad-CAM)
    heatmap = np.random.rand(224, 224)

    # Overlay heatmap on image
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis("off")

    # Save to static folder
    filename = f"heatmap_{uuid.uuid4().hex}.png"
    output_path = f"backend/static/{filename}"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return f"static/{filename}"  # relative path for frontend
