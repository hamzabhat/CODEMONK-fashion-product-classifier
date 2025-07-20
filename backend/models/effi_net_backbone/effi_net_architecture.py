
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class MultiTaskEfficientNet(nn.Module):
    """
    A multitask learning model using a pre-trained EfficientNetV2-S as a backbone.
    It has four separate classification heads to predict color, type, season, and gender.
    """

    def __init__(self, num_colors, num_types, num_seasons, num_genders):
        super().__init__()
        # Load pre-trained backbone with the latest recommended weights
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the number of input features for the classifier
        n_features = self.backbone.classifier[1].in_features
        # Replace the classifier with an Identity layer to get the features
        self.backbone.classifier = nn.Identity()

        # Define separate heads for each task, with Dropout for regularization
        self.color_head = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, num_colors))
        self.type_head = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, num_types))
        self.season_head = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, num_seasons))
        self.gender_head = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, num_genders))

    def forward(self, x):
        # Pass input through the backbone to get shared features
        features = self.backbone(x)
        # Pass features through each head to get task-specific outputs
        return {
            'color': self.color_head(features),
            'product_type': self.type_head(features),
            'season': self.season_head(features),
            'gender': self.gender_head(features)
        }