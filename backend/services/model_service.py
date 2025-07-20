import torch
import json
from PIL import Image
# Assuming your file structure is backend/services/model_service.py
from backend.utils.preprocessing import preprocess_image
from backend.utils.explainability import generate_heatmap
from backend.models.effi_net_backbone.effi_net_architecture import MultiTaskEfficientNet

# --- Load label encoders ---
with open("backend/models/encodings/label_encoders.json", "r") as f:
    label_encoders = json.load(f)

# --- Get number of classes from encoders ---
num_colors = len(label_encoders['baseColour'])
num_types = len(label_encoders['articleType'])
num_seasons = len(label_encoders['season'])
num_genders = len(label_encoders['gender'])

# --- Load trained model ---
model = MultiTaskEfficientNet(
    num_colors=num_colors,
    num_types=num_types,
    num_seasons=num_seasons,
    num_genders=num_genders
)
model.load_state_dict(torch.load("backend/models/checkpoints/model_efficientnet.pth", map_location="cpu"))
model.eval()

# --- Prediction + heatmap generation function ---
def get_predictions_and_heatmap(image_path):
    img_tensor, pil_image = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))

    # --- Convert numeric predictions to class labels ---
    # ⭐ FIX #1: The model output key is 'product_type', not 'type'.
    color_idx = outputs['color'].argmax(dim=1).item()
    type_idx = outputs['product_type'].argmax(dim=1).item()
    season_idx = outputs['season'].argmax(dim=1).item()
    gender_idx = outputs['gender'].argmax(dim=1).item()

    color_pred = label_encoders['baseColour'][str(color_idx)]
    type_pred = label_encoders['articleType'][str(type_idx)]
    season_pred = label_encoders['season'][str(season_idx)]
    gender_pred = label_encoders['gender'][str(gender_idx)]

    # --- Generate heatmap for visualization ---
    heatmap_path = generate_heatmap(model, img_tensor, pil_image)

    # --- Return the final dictionary ---
    return {
        "color": color_pred,
        # ⭐ FIX #2: The key in the response must match what the frontend expects.
        "product_type": type_pred,
        "season": season_pred,
        "gender": gender_pred,
        "heatmap_path": heatmap_path
    }