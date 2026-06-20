import os
from roboflow import Roboflow

def fetch_enterprise_data():
    print("[INFO] Connecting to Roboflow Universe...")
    
    # 1. Initialize the client (You will replace 'YOUR_API_KEY' with your actual key)
    rf = Roboflow(api_key="3O7Ub5wy8LC9AQKTxWeh")
    
    # 2. Target a specific public Nightshade Pest dataset 
    # (Using a robust public YOLOv8 pest detection project)
    project = rf.workspace("tondi").project("pest-detection-yolov8")
    version = project.version(3)
    
    # 3. Set the download destination to your custom folder
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/agrisustain-pests-v1"))
    
    print(f"[INFO] Downloading dataset into: {target_path}")
    
    # 4. Execute the download
    dataset = version.download("yolov8", location=target_path)
    
    print("[INFO] Enterprise data successfully injected into the architecture!")

if __name__ == "__main__":
    fetch_enterprise_data()