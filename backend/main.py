from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
load_model = keras.models.load_model
from fastapi.middleware.cors import CORSMiddleware
import os 
from io import BytesIO
import uvicorn
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from datetime import datetime, timezone
from firebase_config import db 
import uuid
import shutil
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
# CORS Configuration (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify domains instead of "*"
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# # Absolute path to your model
# MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\plant_model.keras"

# # Verify path exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# # Load the model
# try:
#     model = load_model(MODEL_PATH)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     raise






# MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\potato_leaf_disease_model.keras"
# # MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\plant_model.keras"
# # MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\tea_leaf_disease_model.keras"
# model = load_model(MODEL_PATH)
# print("Model loaded successfully!")

# CLASS_NAMES = [
#     "Early Blight", "Late Blight", "Healthy"
# ]
# # CLASS_NAMES = ['1. Tea algal leaf spot', '2. Brown Blight',
# #             '3. Gray Blight', '4. Helopeltis', '5. Red spider', 
# #              '6. Green mirid bug', '7. Healthy leaf', '8.Tea red leaf spot'
# #              ]
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = model.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     if confidence < 0.7:  # Threshold for rejection
#         return {"class": "Unknown", "message": "Not a valid leaf image."}
    
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=8000)













# def preprocess_image(image_bytes: bytes) -> np.ndarray:
#     """Convert uploaded image to model-ready format"""
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = image.resize((224, 224))  # Adjust if your model expects different dimensions
#         image_array = np.array(image) / 255.0  # Normalize pixel values
#         return np.expand_dims(image_array, axis=0)  # Add batch dimension
#     except UnidentifiedImageError:
#         raise ValueError("Invalid image format (supported: JPG, PNG, etc.)")
#     except Exception as e:
#         raise RuntimeError(f"Image processing failed: {str(e)}")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """Handle image upload and return prediction"""
#     try:
#         # Validate file
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
#         contents = await file.read()
#         if len(contents) == 0:
#             raise HTTPException(status_code=400, detail="Empty file uploaded")
        
#         # Process and predict
#         image_tensor = preprocess_image(contents)
#         prediction = model.predict(image_tensor)[0]
        
#         # Format results
#         predicted_index = np.argmax(prediction)
#         confidence = float(prediction[predicted_index])
#         predicted_class = CLASS_NAMES[predicted_index]
        
#         logger.info(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        
#         return JSONResponse(
#             content={
#                 "class": predicted_class,
#                 "confidence": confidence
#             }
#         )
        
#     except HTTPException:
#         raise  # Re-raise FastAPI HTTP exceptions
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)  # Listen on all network interfaces




#   some notes: -->   -->   -->  -->


#   #To run this FastAPI application, save this code in a file named `main.py`. aome 
# to run the server, use the command: uvicorn main:app --reload
# then you can test the API by sending a POST request to http://localhost:8000/predict with an image file
# or http://127.0.0.1:8000/docs
#  to enter into the backend folder inside the git bash : cd "C:/Users/ANISH BHUIN/OneDrive/Desktop/jbooks/backend"
# for command  prompt :::  uvicorn main:app --host 0.0.0.0 --port 8000






# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
# import os 
# from io import BytesIO
# import uvicorn
# from PIL import Image, UnidentifiedImageError
# import numpy as np
# import io
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the model
# # In your FastAPI app where you load the model:
# MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\tea_leaf_disease_model.keras"
# def sparse_focal_loss(gamma=2.0, alpha=0.25):
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.int32)
#         y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # convert inside the loss
#         y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
#         cross_entropy = -y_true * tf.math.log(y_pred)
#         weight = alpha * tf.pow(1 - y_pred, gamma)
#         loss = weight * cross_entropy
#         return tf.reduce_sum(loss, axis=1)
#     return loss
# try:
#     # model = load_model(MODEL_PATH)
#     model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'loss': sparse_focal_loss(alpha=0.25, gamma=2.0)})
#     logger.info("Model loaded successfully!")
# except Exception as e:
#     logger.error(f"Error loading model: {str(e)}")
#     raise

# CLASS_NAMES = [
#     '1. Tea algal leaf spot', 
#     '2. Brown Blight',
#     '3. Gray Blight', 
#     '4. Helopeltis', 
#     '5. Red spider', 
#     '6. Green mirid bug', 
#     '7. Healthy leaf', 
#     '8. Tea red leaf spot'
# ]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data):
#     try:
#         image = Image.open(io.BytesIO(data)).convert("RGB")
#         image = image.resize((224, 224))  # Ensure this matches model's expected input
#         image_array = np.array(image)
        
#         # Normalize if your model expects normalized inputs
#         # (check how your model was trained)
#         image_array = image_array / 255.0  
        
#         return image_array
#     except Exception as e:
#         logger.error(f"Error processing image: {str(e)}")
#         raise HTTPException(status_code=400, detail="Invalid image file")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess the image
#         image_data = await file.read()
#         image_array = read_file_as_image(image_data)
        
#         # Add batch dimension
#         input_tensor = np.expand_dims(image_array, axis=0)
        
#         # Make prediction
#         predictions = model.predict(input_tensor)
#         predicted_class_idx = np.argmax(predictions[0])
#         predicted_class = CLASS_NAMES[predicted_class_idx]
#         confidence = float(np.max(predictions[0]))
        
#         # Set a confidence threshold
#         # if confidence < 0.5:  # Adjust this threshold as needed
#         #     return {
#         #         "class": "Uncertain", 
#         #         "confidence": confidence,
#         #         "message": "Low confidence prediction"
#         #     }
        
#         return {
#             'class': predicted_class, 
#             'confidence': confidence,
#             'all_predictions': {name: float(pred) for name, pred in zip(CLASS_NAMES, predictions[0])}
#         }
        
#     except UnidentifiedImageError:
#         raise HTTPException(status_code=400, detail="Invalid image file")
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=8000)











MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\potato_leaf_disease_model_effnet.keras"
model = load_model(MODEL_PATH)
print("model loaded successfully!")

CLASS_NAMES = [
    "Early Blight", "Late Blight", "Healthy"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # Resize the image to 224x224
    # image = Image.fromarray(image).resize((224, 224))
    image = Image.fromarray(image).resize((300, 300))
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # Push result to Firebase
    timestamp = datetime.now(timezone.utc).isoformat()
    data = {
        "timestamp": timestamp,
        "disease": predicted_class,
        "confidence": float(confidence),
    }
    db.reference("predictions").push(data)
    if confidence < 0.7:  # Threshold for rejection
        return {"class": "Unknown", "message": "Not a valid leaf image."}
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)









                            #    code for LIGHTGBM and EfficientNetV2B3
                            #    code for LIGHTGBM and EfficientNetV2B3
                            #    code for LIGHTGBM and EfficientNetV2B3
                            #    code for LIGHTGBM and EfficientNetV2B3



# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import joblib

# # Correct imports for TensorFlow/Keras
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, preprocess_input

# # Init FastAPI
# app = FastAPI()

# # Load LGBM Model
# MODEL_PATH = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\notebook\lgbm_plant_disease_model.pkl"
# model = joblib.load(MODEL_PATH)
# print("✅ LightGBM model loaded.")

# # Load EfficientNetV2B4 Feature Extractor
# feature_extractor = EfficientNetV2B3(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
# print("✅ EfficientNetV2B4 loaded.")

# # Class Names
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return {"message": "Server running."}

# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).convert("RGB")
#     image = image.resize((224, 224))
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, axis=0)

#     # Extract features
#     features = feature_extractor.predict(img_batch)

#     # Predict
#     predictions = model.predict_proba(features)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = float(np.max(predictions[0]))

#     if confidence < 0.7:
#         return {"class": "Unknown", "message": "Not a valid leaf image."}

#     return {
#         "class": predicted_class,
#         "confidence": confidence
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
