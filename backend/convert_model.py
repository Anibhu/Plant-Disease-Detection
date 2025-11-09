
import os
from keras import models

# Get the absolute path to the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "notebook", "plant_model.keras")

# Debug output
print(f"Current directory: {current_dir}")
print(f"Looking for model at: {model_path}")

# Verify path exists
if not os.path.exists(model_path):
    print("\nAvailable files in notebook/:")
    print(os.listdir(os.path.join(current_dir, "..", "notebook")))
    raise FileNotFoundError("Model file not found at above path")

# Load and export
model = models.load_model(model_path)
model.export(os.path.join(current_dir, "saved_model", "plant_disease_model"))
print("\n✅ Success! Model converted to SavedModel format for TF Serving.")






#                            :: Note :: 
# ✅ Option B: Use Task Scheduler + Batch Script (Auto-start at boot)
# 💡 Here's how to do it:
# Create a .bat file to run your FastAPI server.

# start_server.bat:

# bat
# Copy code
# cd C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\backend
# uvicorn main:app --host 0.0.0.0 --port 8000
# Open Task Scheduler

# Search Task Scheduler in Windows.

# Click Create Basic Task.

# Name: Start FastAPI Server

# Trigger: When I log on

# Action: Start a program

# Browse: Select your start_server.bat file

# Finish.

# ✅ Now your FastAPI server starts automatically every time your PC boots/logs in — no need to open Bash manually again.

# ✅ Option C (Alternative): Use Docker + Always running container
# Let me know if you're familiar with Docker, we can wrap FastAPI in a container that auto-restarts. But Option B is easier for now.

# ✅ 2. How Raspberry Pi can keep taking pictures and sending them
# You can schedule the Raspberry Pi script to:

# Run at boot

# Repeat every X minutes

# Or both!

# ✅ Option A: Run your predict_leaf.py on boot (Raspberry Pi)
# Add this line to your crontab:

# bash
# Copy code
# crontab -e
# At the end of the file, add:

# bash
# Copy code
# @reboot python3 /home/pi/predict_leaf.py
# 🔁 You can also use a loop inside the script to repeat every X seconds:

# python
# Copy code
# import time
# while True:
#     capture_and_send_image()
#     time.sleep(60)  # every 1 minute
# Or use cron to repeat every 5 minutes:

# bash
# Copy code
# */5 * * * * python3 /home/pi/predict_leaf.py