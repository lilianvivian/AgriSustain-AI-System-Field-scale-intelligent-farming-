import os
from ultralytics import YOLO

def train_pest_detector():
    # 1. Load an enterprise-grade pre-trained YOLOv8 model backbone
    # 'yolov8m.pt' (medium) provides an excellent balance of speed and accuracy for edge devices
    model = YOLO("yolov8m.pt")

    # 2. Define the path to your data.yaml file
    # This automatically finds where the script is located, then maps to the data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(script_dir, "../../data/agrisustain-pests-v1/data.yaml"))

    print(f"[INFO] Starting training using configuration at: {yaml_path}")

    # 3. Kick off training with advanced data augmentations to target camouflage and deformation
    results = model.train(
        data=yaml_path,
        epochs=50,          # 50 epochs ensures convergence for complex multi-object tasks
        imgsz=640,          # Enterprise standard image resolution for training
        batch=16,           # Change to 8 or 4 if your GPU runs out of VRAM
        device=0,           # Set to 0 to use your Lightning.ai GPU acceleration
        project="../../saved_models", # Save output here
        name="pest_detection_run",
        save=True,          # Secure checkpoints during training
        
        # === DATA AUGMENTATION HYPERPARAMETERS ===
        degrees=15.0,       # Random rotation +/- 15 degrees (handles flexible pest orientation)
        scale=0.5,          # Scale images up/down by 50% (handles scale variations in field view)
        fliplr=0.5,         # Flip left-right with 50% probability
        flipud=0.2,         # Flip upside-down with 20% probability (helps orientation variety)
        hsv_h=0.015,        # Subtle shift in image hue
        hsv_s=0.7,          # Heavy saturation shift (breaks camouflage for green pests like Caterpillars)
        hsv_v=0.4,          # Heavy brightness value shift (helps extract features from dark soil/Earthworms)
        mosaic=1.0          # Stitches 4 training images together to boost micro-scale detection
    )
    
    print("[INFO] Training complete! Best weights saved to saved_models/pest_detection_run/weights/best.pt")

if __name__ == "__main__":
    train_pest_detector()