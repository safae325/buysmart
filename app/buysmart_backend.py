import cv2
import os
from app.utils import match_image, get_color_by_criteria
from ultralytics import YOLO
import sqlite3

# Load YOLO model
model = YOLO("models/yolov8n.pt")  # Update path to reflect the new location

DB_PATH = "database/products.db"  # Path to your SQLite database

def fetch_product_info(barcode):
    """
    Fetch product information from the database by barcode.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT code, product_name, nutriscore_grade, ecoscore_grade, price, palm_oil, 
               price_per_kg, recommended_by, quantity, labels, sugars, ingredients
        FROM products WHERE code = ?
    """
    cursor.execute(query, (barcode,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "code": row[0],
            "name": row[1],
            "nutriscore": row[2],
            "ecoscore": row[3],
            "price": row[4],
            "palm_oil": row[5],
            "price_per_kg": row[6],
            "recommended_by": row[7],
            "quantity": row[8],
            "labels": row[9],
            "sugars": row[10],
            "ingredients": row[11]
        }
    else:
        return None
import os

def crop_and_save_products(image_path, detected_products, output_folder="cropped_products"):
    """
    Crop detected products from the image and save them as separate files.

    Args:
        image_path (str): Path to the original image.
        detected_products (list): List of detected products with bounding boxes.
        output_folder (str): Folder to save the cropped images.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Crop and save each detected product
    for i, product in enumerate(detected_products):
        x1, y1, x2, y2 = product["bbox"]

        # Crop the region of interest
        cropped_image = image[y1:y2, x1:x2]

        # Generate a unique filename
        product_name = product["product_info"]["name"].replace(" ", "_")
        output_path = os.path.join(output_folder, f"{i}_{product_name}.jpg")

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped product saved to {output_path}")



def deduplicate_detections(detections):
    """
    Deduplicate detections by retaining the one with the highest confidence per barcode.
    """
    unique_detections = {}
    for detection in detections:
        barcode = detection["barcode"]
        if barcode not in unique_detections or detection["confidence"] > unique_detections[barcode]["confidence"]:
            unique_detections[barcode] = detection
    return list(unique_detections.values())

def detect_and_process(image_path, criterion):
    """
    Detect products, match them with database entries, and process them.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    # Run YOLO detection
    results = model(image_path, save=False)  # Save the output image with bounding boxes
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    detected_products = []
    all_prices = []

    print(f"Detected {len(detections)} bounding boxes.")  # Debugging
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4])
        confidence = confidences[i]
        print(f"Detection {i}: bbox=({x1}, {y1}, {x2}, {y2}), Confidence: {confidence}")

        if confidence > 0.2 :  # Lower threshold to include more detections
            cropped_image = image[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Match the cropped image with stored product images
            matched_barcode = match_image(cropped_rgb)
            print(f"Matched Barcode for Detection {i}: {matched_barcode}")

            if matched_barcode:
                product_info = fetch_product_info(matched_barcode)
                if product_info:
                    # Collect prices for gradient calculation
                    if product_info["price"] is not None:
                        all_prices.append(float(product_info["price"]))

                    detected_products.append({
                        "bbox": [x1, y1, x2, y2],
                        "barcode": matched_barcode,
                        "confidence": round(float(confidence), 2),
                        "product_info": {
                            "name": product_info.get("name", f"Product with barcode {matched_barcode}"),
                            "description": product_info.get("ingredients"),
                            "price": product_info["price"],
                            "quantity": product_info["quantity"],
                            "price_per_kg": product_info["price_per_kg"],
                            "sugars": product_info["sugars"],
                            "nutriscore": product_info.get("nutriscore"),
                            "ecoscore": product_info.get("ecoscore"),
                            "palm_oil": product_info.get("palm_oil"),
                            "recommended_by": product_info.get("recommended_by"),
            
           
                        }
                    })

    print(f"Before Deduplication: {len(detected_products)} products")
    deduplicated_products = deduplicate_detections(detected_products)
    print(f"After Deduplication: {len(deduplicated_products)} products")

    # Assign colors based on the selected criterion
    for product in deduplicated_products:
        product["color"] = get_color_by_criteria(
            nutriscore=product["product_info"].get("nutriscore"),
            ecoscore=product["product_info"].get("ecoscore"),
            price=product["product_info"].get("price"),
            palm_oil=product["product_info"].get("palm_oil"),
            criterion=criterion,
            all_prices=all_prices
        )


    return deduplicated_products

