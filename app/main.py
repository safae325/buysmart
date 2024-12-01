import os
from fastapi import FastAPI, File, UploadFile, Form
from app.buysmart_backend import detect_and_process

app = FastAPI()
UPLOAD_FOLDER = "uploads/"
FOLDER_PATH = "images/"  # Path to your stored product images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/detect-products/")
async def detect_products(
    image: UploadFile = File(...),
    criterion: str = Form(...)
):
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(image.file.read())

    try:
        detected_products = detect_and_process(file_path, criterion)
        return {
            "status": "success",
            "criterion": criterion,
            "products": detected_products
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)



@app.get("/")
def home():
    """
    Health check endpoint for the API.
    """
    return {"message": "BuySmart API is running."}
