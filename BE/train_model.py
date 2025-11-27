"""
🏀 BASKETBALL DETECTION - YOLO11 Training Script (Refactored)
=============================================================
This script handles the training process for the basketball detection model.
It includes:
1. Automatic dataset validation.
2. GPU hardware verification.
3. Advanced interruption handling (Ctrl+C).
4. Custom logging and metrics visualization.
5. Optimized hyperparameters for basketball motion tracking.

Author: Refactored by Gemini
Target: 90%+ mAP@50 on 5 classes
"""

import sys
import signal
import time
import yaml
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION CLASS
# ==============================================================================
class Config:
    """
    Centralizes all configuration parameters, paths, and hyperparameters.
    Modify this section to tune the training process.
    """
    # --- Paths ---
    PROJECT_NAME = "basketball_training"
    RUN_NAME = "yolo11s_5classes"
    DATASET_DIR = Path("basketball-detection-srfkd-1")
    DATA_YAML = "data_basketball.yaml"
    BASE_MODEL = "yolo11s.pt"  # Starting point (Pre-trained COCO)
    
    # --- Checkpoint Handling ---
    RESUME_PATH = Path(f"{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
    
    # --- Hardware & System ---
    WORKERS = 0        # Set to 0 for Windows compatibility to avoid multiprocessing errors
    DEVICE = 0         # GPU Index (0 for the first GPU)
    SEED = 42          # Fixed seed for reproducibility
    
    # --- Core Training Hyperparameters ---
    EPOCHS = 200       # Total number of training epochs
    BATCH_SIZE = 8     # Batch size (Adjust based on VRAM, 8 is good for 6GB VRAM)
    IMG_SIZE = 640     # Input image resolution
    PATIENCE = 30      # Early stopping patience (epochs without improvement)
    SAVE_PERIOD = 5    # Save heavy checkpoints every X epochs
    OPTIMIZER = 'AdamW'
    
    # --- Learning Rate Strategy ---
    LR0 = 0.01         # Initial learning rate (SGD=1E-2, Adam=1E-3)
    LRF = 0.001        # Final learning rate (lr0 * lrf)
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3.0
    COS_LR = True      # Use Cosine LR scheduler
    
    # --- Loss Function Weights ---
    # Adjusted to prioritize bounding box accuracy over classification
    BOX_GAIN = 7.5     # Box loss gain
    CLS_GAIN = 0.5     # Class loss gain
    DFL_GAIN = 1.5     # Distribution Focal Loss gain
    
    # --- Data Augmentation (Optimized for Sports/Motion) ---
    # Heavy augmentation helps YOLO generalize on limited datasets
    AUGMENTATION = {
        'hsv_h': 0.015,     # HSV-Hue adjustment
        'hsv_s': 0.7,       # HSV-Saturation adjustment
        'hsv_v': 0.4,       # HSV-Value adjustment
        'degrees': 10.0,    # Rotation (+/- deg)
        'translate': 0.1,   # Translation (+/- fraction)
        'scale': 0.6,       # Scale gain (+/- gain)
        'shear': 2.0,       # Shear angle (+/- deg) - Important for basket perspective
        'perspective': 0.0005, # Perspective warp
        'flipud': 0.0,      # Vertical flip (Disabled: gravity matters)
        'fliplr': 0.5,      # Horizontal flip (Enabled: courts are symmetric)
        'mosaic': 1.0,      # Mosaic (Probability)
        'mixup': 0.15,      # Mixup (Probability) - Helps with player overlap
        'copy_paste': 0.1,  # Segment copy-paste (Probability)
        'erasing': 0.4,     # Random erasing (Probability) - Simulates occlusion
        'auto_augment': 'randaugment', # Use RandAugment policy
    }

# ==============================================================================
# 2. LOGGING UTILITIES
# ==============================================================================
class TrainingLogger:
    """
    Handles console output formatting and progress tracking.
    """
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None

    def start_session(self):
        """Called when training session begins."""
        self.start_time = time.time()
        print("\n" + "🚀 " * 20)
        print(f"STARTING TRAINING SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🚀 " * 20 + "\n")

    def start_epoch(self, current, total):
        """Called at the start of each epoch."""
        self.epoch_start_time = time.time()
        print(f"\n{'─'*60}")
        print(f"🔄 EPOCH {current}/{total} STARTED")
        print(f"{'─'*60}")

    def log_metrics(self, epoch, total_epochs, metrics):
        """Parses and prints metrics after an epoch."""
        elapsed = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.start_time
        
        # Robust metric extraction (handles both Dict and Class interfaces from YOLO)
        def get_val(key_chain, default=0):
            val = metrics
            for k in key_chain:
                val = getattr(val, k, val.get(k) if isinstance(val, dict) else default)
            return val

        # Handle different YOLO version outputs
        if isinstance(metrics, dict):
             # Dictionary access
            map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50', 0))
            map5095 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95', 0))
            prec = metrics.get('metrics/precision(B)', metrics.get('metrics/precision', 0))
            rec = metrics.get('metrics/recall(B)', metrics.get('metrics/recall', 0))
        else:
             # Object attribute access
            map50 = getattr(metrics, 'map50', 0)
            map5095 = getattr(metrics, 'map', 0)
            prec = getattr(getattr(metrics, 'box', metrics), 'mp', 0)
            rec = getattr(getattr(metrics, 'box', metrics), 'mr', 0)

        # Visual Output
        print(f"\n📊 EPOCH {epoch} METRICS:")
        print(f"   • mAP@50:    {map50:.4f}")
        print(f"   • mAP@50-95: {map5095:.4f}")
        print(f"   • Precision: {prec:.4f}")
        print(f"   • Recall:    {rec:.4f}")
        print(f"\n⏱️  Timing: {elapsed/60:.1f} min/epoch | Total: {total_elapsed/3600:.1f} hours")

        # Progress Bar
        progress = epoch / total_epochs
        filled_len = int(40 * progress)
        bar = "█" * filled_len + "░" * (40 - filled_len)
        print(f"   Progress: [{bar}] {progress*100:.1f}%")

# Global logger instance for callbacks
logger = TrainingLogger()

# ==============================================================================
# 3. HELPER CLASSES
# ==============================================================================
class DatasetValidator:
    """
    Ensures the dataset structure and YAML configuration are correct before starting.
    """
    @staticmethod
    def validate():
        print("🔍 Validating dataset configuration...")
        
        yaml_path = Config.DATASET_DIR / Config.DATA_YAML
        
        # Check YAML existence
        if not yaml_path.exists():
            print(f"❌ Critical Error: {Config.DATA_YAML} not found at {yaml_path.absolute()}")
            sys.exit(1)
            
        # Parse YAML
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Verify Paths
        status_ok = True
        for split_key in ['train', 'val', 'test']:
            if split_key in data:
                # Construct absolute path to image directory
                img_path = Config.DATASET_DIR / data[split_key]
                if not img_path.exists():
                    print(f"❌ Missing {split_key} directory: {img_path}")
                    status_ok = False
                else:
                    count = len(list(img_path.glob('*.*')))
                    print(f"✅ {split_key.upper()}: Found {count} images.")
        
        if not status_ok:
            print("\n⛔ Dataset validation failed. Please check paths in data.yaml.")
            sys.exit(1)
            
        print(f"📋 Classes: {data.get('names', 'Unknown')}")
        print("✅ Dataset validation passed!\n")
        return str(yaml_path.absolute())

class InterruptionHandler:
    """
    Handles CTRL+C signals to save state gracefully.
    """
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        
    def _handle_signal(self, sig, frame):
        print("\n\n" + "⚠️ " * 20)
        print("TRAINING INTERRUPTED BY USER (SIGINT)")
        print("⚠️ " * 20)
        print(f"📁 Last checkpoint is saved at: {Config.RESUME_PATH}")
        print("🔄 To resume, simply run this script again.")
        sys.exit(0)

# ==============================================================================
# 4. TRAINING LOGIC
# ==============================================================================
class TrainingSession:
    def __init__(self):
        self.resume_training = Config.RESUME_PATH.exists()
        self.interruption_handler = InterruptionHandler()
        
    def _check_hardware(self):
        print(f"{'='*60}")
        print("🖥️  HARDWARE DIAGNOSTICS")
        print(f"{'='*60}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"🚀 GPU: {gpu_name}")
            print(f"💾 VRAM: {vram_total:.1f} GB Total | {vram_allocated:.2f} GB Allocated")
        else:
            print("❌ CUDA NOT DETECTED. Training on CPU is extremely slow.")
            user_input = input("   Continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                sys.exit(0)
        print(f"{'='*60}\n")

    def _setup_callbacks(self, model):
        """Attaches custom logging callbacks to the YOLO model."""
        def on_train_epoch_start(trainer):
            logger.start_epoch(trainer.epoch + 1, trainer.epochs)
            
        def on_train_epoch_end(trainer):
            logger.log_metrics(trainer.epoch + 1, trainer.epochs, trainer.metrics)
            
        def on_train_start(trainer):
            logger.start_session()
            
        model.add_callback('on_train_start', on_train_start)
        model.add_callback('on_train_epoch_start', on_train_epoch_start)
        model.add_callback('on_train_epoch_end', on_train_epoch_end)

    def run(self):
        # 1. Validate Dataset
        yaml_path = DatasetValidator.validate()
        
        # 2. Check Hardware
        self._check_hardware()
        
        # 3. Load Model
        print("📦 Loading YOLO Model...")
        if self.resume_training:
            print(f"🔄 Resuming from checkpoint: {Config.RESUME_PATH}")
            model = YOLO(str(Config.RESUME_PATH))
        else:
            print(f"🆕 Starting fresh from base model: {Config.BASE_MODEL}")
            model = YOLO(Config.BASE_MODEL)
            
        # 4. Setup Callbacks
        self._setup_callbacks(model)
        
        # 5. Build Training Arguments
        # Merging core config with augmentation settings
        train_args = {
            'data': yaml_path,
            'project': Config.PROJECT_NAME,
            'name': Config.RUN_NAME,
            'epochs': Config.EPOCHS,
            'batch': Config.BATCH_SIZE,
            'imgsz': Config.IMG_SIZE,
            'device': Config.DEVICE,
            'workers': Config.WORKERS,
            'optimizer': Config.OPTIMIZER,
            'patience': Config.PATIENCE,
            'save': True,
            'save_period': Config.SAVE_PERIOD,
            'cache': 'disk', # Reduces RAM usage
            'exist_ok': True,
            'pretrained': True,
            'verbose': True, # Keep internal verbose logging
            'seed': Config.SEED,
            'cos_lr': Config.COS_LR,
            'close_mosaic': 15, # Disable mosaic for last 15 epochs for finer detail
            
            # Hyperparameters
            'lr0': Config.LR0,
            'lrf': Config.LRF,
            'momentum': Config.MOMENTUM,
            'weight_decay': Config.WEIGHT_DECAY,
            'warmup_epochs': Config.WARMUP_EPOCHS,
            'box': Config.BOX_GAIN,
            'cls': Config.CLS_GAIN,
            'dfl': Config.DFL_GAIN,
            
            # Flatten augmentation dict into arguments
            **Config.AUGMENTATION,
            
            # Misc
            'resume': self.resume_training,
            'val': True,
            'plots': True,
            'multi_scale': True # Helps detection at different distances
        }
        
        # 6. Start Training
        print(f"\n🎯 Target: {Config.EPOCHS} Epochs | Batch Size: {Config.BATCH_SIZE}")
        print("⏳ Initializing training pipeline (this might take a minute)...")
        
        try:
            results = model.train(**train_args)
            self._finalize(model)
        except Exception as e:
            print(f"\n❌ FATAL ERROR DURING TRAINING: {e}")
            raise e

    def _finalize(self, model):
        """Final validation and success message."""
        print("\n" + "🎉" * 20)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("🎉" * 20)
        
        print("\n🔍 Running Final Validation...")
        metrics = model.val()
        
        # Check if we hit the 90% target (Approximate check)
        try:
            map50 = metrics.box.map50
            if map50 > 0.90:
                print(f"\n✅ SUCCESS: Model achieved >90% mAP@50 ({map50:.2f})")
            else:
                print(f"\n⚠️ NOTE: Model reached {map50:.2f} mAP@50. Consider more epochs or more data.")
        except:
            pass
            
        print(f"📁 Final weights saved at: {Config.PROJECT_NAME}/{Config.RUN_NAME}/weights/best.pt")

# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    session = TrainingSession()
    session.run()