from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"❌ Cannot identify image file: {image_path}")

    return transform(image), image  # (tensor, PIL)
