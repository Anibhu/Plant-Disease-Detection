import firebase_admin
from firebase_admin import credentials, db

# Load your Firebase service account key JSON
cred = credentials.Certificate("plant-disease-detector-1d5e1-firebase-adminsdk-fbsvc-5b0928c2d7.json")

# Initialize Firebase app with Realtime Database URL
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://plant-disease-detector-1d5e1-default-rtdb.firebaseio.com/ML"
})