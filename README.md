Animal Detection V2

YOLO-based real-time wildlife detection system
Detects six animal species—buffaloes, deer, elephants, rhinos, tigers, wild boars—from images or live camera feeds.
Includes dataset preparation tools, YOLO training scripts, and live inference with audio alerts.

Setup
Create and Activate a Python Environment
python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

Dataset Preparation

Split the flat dataset into the Ultralytics YOLO directory format:

python prepare_dataset.py

Optional Arguments:

--ratios 0.7 0.2 0.1 → custom train/val/test split

--seed 123 → fixed shuffling

--move → move files instead of copying

Output directory:

datasets/animals/

 Training the YOLO Model

Fine-tune a YOLO checkpoint:

python train_yolo.py --model yolov8n.pt --device 0 --epochs 100

Useful Flags

--batch 16 → batch size

--imgsz 640 → training resolution

--project runs/animals

--name yolov8n-animals

--resume → continue previous training

Outputs

Trained weights are saved at:

runs/animals/<run-name>/weights/{best.pt, last.pt}

🎥 Real-Time Detection + Audio Alerts

Run inference from webcam/video and trigger an alert on detection:

python detect_and_alert.py --source 0 --audio path\to\alert.wav

Arguments

--model runs/animals/yolov8n-animals3/weights/best.pt

--conf 0.35 → confidence threshold

--cooldown 2.0 → minimum delay between alerts

--device 0 → GPU (cpu fallback available)

If no model path is provided, the script automatically selects the latest weights from:

runs/animals/**/weights/

 Dependencies

Installed via:

pip install -r requirements.txt


Includes:

ultralytics → YOLO training & inference

opencv-python → video I/O

playsound → audio notifications

polars → fast logging backend

Make sure NVIDIA drivers + CUDA runtime are installed if training on GPU.

 Useful Tips

Check model performance, predictions, and logs in:

runs/animals/


Validate on test set:

yolo val


For offline video analysis:

python detect_and_alert.py --source path\to\video.mp4


Use --move carefully in dataset script (files will be permanently relocated).
