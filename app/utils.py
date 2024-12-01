import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import transforms

# Pretrained ResNet model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # Remove classification head
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to precomputed embeddings
EMBEDDINGS_PATH = "image_embeddings.pkl"

def extract_features(image):
    """
    Extract deep learning features from the given image using ResNet.
    """
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        return model(input_tensor).squeeze().numpy()

def match_image(cropped_image):
    """
    Match cropped product image with precomputed stored image embeddings.
    """
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)

    cropped_features = extract_features(cropped_image)
    best_match = None
    highest_similarity = 0
    similarity_threshold = 0.7  # Lower threshold for testing

    for image_path, data in embeddings.items():
        similarity = cosine_similarity([cropped_features], [data["features"]])[0][0]
        print(f"Similarity with {image_path}: {similarity}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = data["barcode"]

    if highest_similarity >= similarity_threshold:
        print(f"Best match: {best_match} with similarity {highest_similarity}")
        return best_match
    else:
        print("No reliable match found for this image.")
        return None


def get_color_by_criteria(nutriscore, ecoscore, price, palm_oil, criterion, all_prices=None):
    """
    Assign RGB color based on the selected criterion:
    - Nutriscore/Ecoscore: Fixed mappings.
    - Price: Dynamic gradient based on the range of prices.
    """
    # Palm oil handling: Fixed colors
    if criterion ==palm_oil == "with":
        return [0, 0, 0]  # Black for products containing palm oil
    elif palm_oil == "without" and criterion == "palm_oil":
        return [0, 255, 0]  # Green for products without palm oil

    # Fixed color mappings for Nutriscore
    if criterion == "nutriscore":
        nutriscore_colors = {"A": [0, 255, 0], "B": [173, 255, 47], "C": [255, 255, 0], "D": [255, 165, 0], "E": [255, 0, 0]}
        return nutriscore_colors.get(nutriscore.upper(), [128, 128, 128])  # Gray for unknown values

    # Fixed color mappings for Ecoscore
    if criterion == "ecoscore":
        ecoscore_colors = {"A": [0, 255, 0], "B": [173, 255, 47], "C": [255, 255, 0], "D": [255, 165, 0], "E": [255, 0, 0]}
        return ecoscore_colors.get(ecoscore.upper(), [128, 128, 128])  # Gray for unknown values

    # Dynamic gradient for Price
    if criterion == "price":
        if not all_prices or len(all_prices) == 1:
            return [0, 255, 0]  # Default to green if no comparison is possible

        min_price = min(all_prices)
        max_price = max(all_prices)
        range_price = max_price - min_price

        if range_price == 0:
            return [0, 255, 0]  # All prices are equal

        # Scale price to 0-1 for dynamic color
        normalized_price = (price - min_price) / range_price
        if normalized_price == 0:
            return [0, 255, 0]  # Green for the cheapest
        elif normalized_price == 1:
            return [255, 0, 0]  # Red for the most expensive
        else:
            # Gradual gradient: Green -> Yellow -> Orange -> Red
            red = int(255 * normalized_price)
            green = int(255 * (1 - normalized_price))
            return [red, green, 0]

    # Default color (gray) for unsupported criteria
    return [128, 128, 128]
