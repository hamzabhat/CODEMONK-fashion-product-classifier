from backend.utils.preprocessing import preprocess_image
from backend.utils.explainability import generate_heatmap
import torch
from PIL import Image
import numpy as np
# ⬇️ Define your class names
COLOR_CLASSES = ['Black', 'White', 'Blue', 'Red']
TYPE_CLASSES = ['Shoes', 'T-shirt', 'Shirt', 'Jeans']
SEASON_CLASSES = ['Summer', 'Winter', 'All']
GENDER_CLASSES = ['Men', 'Women', 'Unisex']

# ⬇️ Load trained model once
model = torch.load("backend/models/model.pth", map_location=torch.device("cpu"))
model.eval()

def get_predictions_and_heatmap(image_path):

    img_tensor, pil_image = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))

    # Extract predictions from model output dict
    color_pred = COLOR_CLASSES[outputs['color'].argmax()]
    type_pred = TYPE_CLASSES[outputs['type'].argmax()]
    season_pred = SEASON_CLASSES[outputs['season'].argmax()]
    gender_pred = GENDER_CLASSES[outputs['gender'].argmax()]

    # Generate Grad-CAM heatmap
    heatmap_filename = generate_heatmap(model, img_tensor, pil_image)

    return {
        "color": color_pred,
        "type": type_pred,
        "season": season_pred,
        "gender": gender_pred,
        "heatmap_path": heatmap_filename
    }
