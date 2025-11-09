"""
🏀 BASKETBALL DETECTION - YOLO11s Training Script
Optimized for GTX 1060 6GB + Windows
Target: 90%+ mAP@50, Precision, Recall on all 5 classes
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import signal
import sys
from datetime import datetime
import time

# ==================== CONFIGURAZIONE ====================
DATASET_PATH = Path("basketball-detection-srfkd-1")
DATA_YAML = "data_basketball.yaml"  # File YAML ottimizzato
BASE_MODEL = "yolo11s.pt"  # Small model per 10-20h training

# Auto-resume se esiste checkpoint
RESUME_PATH = Path("basketball_training/yolo11s_5classes/weights/last.pt")
RESUME_TRAINING = RESUME_PATH.exists()

# ==================== GESTIONE INTERRUZIONI ====================
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    print("\n\n" + "="*60)
    print("⚠️  TRAINING PAUSED BY USER")
    print("="*60)
    print(f"📁 Checkpoint saved at: {RESUME_PATH}")
    print("🔄 To resume, simply run this script again")
    print("   RESUME_TRAINING will auto-detect the checkpoint")
    print("="*60)
    interrupted = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==================== LOGGING ====================
class TrainingLogger:
    def __init__(self):
        self.start_time = None
        self.epoch_start = None
        
    def start_training(self):
        self.start_time = time.time()
        
    def start_epoch(self, epoch, total_epochs):
        self.epoch_start = time.time()
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        print("\n" + "─"*60)
        print(f"🔄 EPOCH {epoch}/{total_epochs}")
        print("─"*60)

    def end_epoch(self, epoch, metrics):
        elapsed = time.time() - self.epoch_start
        total_elapsed = time.time() - self.start_time

        print("\n📊 Metrics:")
        # metrics potrebbe essere dict oppure un oggetto: gestiamo entrambi i casi
        if isinstance(metrics, dict):
            m_map50 = metrics.get('metrics/mAP50', metrics.get('metrics/mAP50(B)', 0))
            m_map5095 = metrics.get('metrics/mAP50-95', metrics.get('metrics/mAP50-95(B)', 0))
            precision = metrics.get('metrics/precision', metrics.get('metrics/precision(B)', 0))
            recall = metrics.get('metrics/recall', metrics.get('metrics/recall(B)', 0))
        else:
            # qualche versione della libreria può ritornare oggetti con attributi
            m_map50 = getattr(getattr(metrics, 'box', metrics), 'map50', getattr(metrics, 'map50', 0))
            m_map5095 = getattr(getattr(metrics, 'box', metrics), 'map', getattr(metrics, 'map', 0))
            precision = getattr(getattr(metrics, 'box', metrics), 'mp', getattr(metrics, 'precision', 0))
            recall = getattr(getattr(metrics, 'box', metrics), 'mr', getattr(metrics, 'recall', 0))

        print(f"  mAP@50:     {m_map50:.4f}")
        print(f"  mAP@50-95:  {m_map5095:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")

        print(f"\n⏱️  Epoch time: {elapsed/60:.1f}m | Total: {total_elapsed/3600:.1f}h")

        # Progress bar visuale: usa total_epochs memorizzato
        total_epochs = getattr(self, 'total_epochs', None) or epoch
        progress = epoch / total_epochs
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"Progress: [{bar}] {progress*100:.1f}%")

logger = TrainingLogger()

# ==================== CALLBACK PERSONALIZZATO ====================
def on_train_epoch_end(trainer):
    """Callback chiamato alla fine di ogni epoca"""
    metrics = trainer.metrics
    epoch = trainer.epoch + 1
    logger.end_epoch(epoch, metrics)

def on_train_start(trainer):
    """Callback chiamato all'inizio del training"""
    logger.start_training()
    print("\n🚀 Training started!")

def on_train_epoch_start(trainer):
    """Callback chiamato all'inizio di ogni epoca"""
    logger.start_epoch(trainer.epoch + 1, trainer.epochs)

# ==================== VERIFICA DATASET ====================
def verify_dataset():
    """Verifica integrità del dataset"""
    print("\n🔍 Verifying dataset...")
    
    yaml_path = DATASET_PATH / DATA_YAML
    if not yaml_path.exists():
        print(f"❌ {DATA_YAML} not found!")
        print(f"   Expected at: {yaml_path.absolute()}")
        print("\n💡 Make sure you have:")
        print("   1. Downloaded the dataset from Roboflow")
        print("   2. Created data_basketball.yaml with correct paths")
        sys.exit(1)
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Verifica directories
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_PATH / data[split]
        lbl_dir = DATASET_PATH / split / 'labels'
        
        if not img_dir.exists():
            print(f"❌ Images directory not found: {img_dir}")
            sys.exit(1)
        if not lbl_dir.exists():
            print(f"❌ Labels directory not found: {lbl_dir}")
            sys.exit(1)
        
        img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        lbl_count = len(list(lbl_dir.glob('*.txt')))
        
        print(f"✅ {split:5s}: {img_count:4d} images, {lbl_count:4d} labels")
    
    print(f"\n📋 Classes: {data['names']}")
    print(f"✅ Dataset verification passed!\n")
    
    return yaml_path

# ==================== TRAINING ====================
def train_model():
    """Funzione principale di training"""
    print("="*60)
    print("🏀 BASKETBALL DETECTION - YOLO11s TRAINING")
    print("   5 Classes: ball, ball-in-basket, player, basket, player-shooting")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Model: {BASE_MODEL}")
    print(f"💾 Dataset: {DATASET_PATH}")
    print(f"🖥️  Hardware: GTX 1060 6GB + 16GB RAM + i7-6700K")
    print(f"🔄 Resume: {'YES ✅' if RESUME_TRAINING else 'NO (Fresh start)'}")
    if RESUME_TRAINING:
        print(f"📂 Checkpoint: {RESUME_PATH}")
    print("="*60)
    
    # Verifica dataset
    yaml_path = verify_dataset()
    
    # ==================== GPU INFO ====================
    print("\n" + "="*60)
    print("🖥️  GPU INFORMATION")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.version.cuda}")
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔥 Current VRAM Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("❌ CUDA not available - Training will be VERY slow on CPU!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    print("="*60)
    
    # ==================== CARICA MODELLO ====================
    print("\n📦 Loading model...")
    if RESUME_TRAINING:
        model = YOLO(str(RESUME_PATH))
        print(f"✅ Loaded checkpoint: {RESUME_PATH}")
    else:
        model = YOLO(BASE_MODEL)
        print(f"✅ Loaded base model: {BASE_MODEL}")
    

    # CALLBACKS 
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    # ==================== PARAMETRI TRAINING ====================
    print("\n" + "="*60)
    print("⚙️  TRAINING PARAMETERS")
    print("="*60)
    
    params = {
        # Core
        'data': str(yaml_path),
        'epochs': 200,              # ↑ Aumentato per 5 classi
        'batch': 8,                # Ottimale per 1060 6GB
        'imgsz': 640,
        'device': 0,
        'patience': 30,             # ↑ Più pazienza per convergenza
        'save': True,
        'save_period': 5,           # Salva ogni 5 epoch
        'cache': 'disk',            # Usa disk cache (più lento ma risparmia RAM)
        'workers': 0,               # ⚠️ WINDOWS: deve essere 0!
        'project': 'basketball_training',
        'name': 'yolo11s_5classes',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'cos_lr': True,             # Cosine learning rate decay
        'close_mosaic': 15,         # Disabilita mosaic negli ultimi 15 epoch
        'amp': True,               
        'fraction': 1.0,            # Usa tutto il dataset
        
        # ==================== LEARNING RATES ====================
        'lr0': 0.01,                # Initial learning rate
        'lrf': 0.001,               # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # ==================== LOSS WEIGHTS ====================
        'box': 7.5,                 # Box loss weight
        'cls': 0.5,                 # Class loss weight
        'dfl': 1.5,                 # DFL loss weight
        
        # ==================== DATA AUGMENTATION ====================
        # Ottimizzato per tracking palla in movimento
        'hsv_h': 0.015,             # HSV-Hue augmentation
        'hsv_s': 0.7,               # HSV-Saturation
        'hsv_v': 0.4,               # HSV-Value
        'degrees': 10.0,            # ↑ Rotation augmentation (più variabilità)
        'translate': 0.1,           # Translation
        'scale': 0.6,               # ↑ Scale augmentation
        'shear': 2.0,               # ↑ Shear (importante per prospettive canestro)
        'perspective': 0.0005,      # ↑ Perspective warp (canestro da angoli diversi)
        'flipud': 0.0,              # NO vertical flip (basket non si ribalta)
        'fliplr': 0.5,              # Horizontal flip (campo simmetrico)
        'mosaic': 1.0,              # Mosaic augmentation
        'mixup': 0.15,              # ↑ Mixup (aiuta con occlusioni)
        'copy_paste': 0.1,          # ↑ Copy-paste augmentation
        'auto_augment': 'randaugment',  # ⭐ RandAugment per variabilità
        'erasing': 0.4,             # ⭐ Random erasing (simula occlusioni)
        
        # ==================== MOTION BLUR SIMULATION ====================
        # YOLO11 non ha motion blur nativo, ma randaugment lo include parzialmente
        
        # ==================== ADVANCED ====================
        'overlap_mask': True,       # Masks can overlap
        'mask_ratio': 4,            # Mask downsample ratio
        'dropout': 0.0,             # Dropout (0 = disabled)
        'val': True,                # Validate during training
        'plots': True,              # Save plots
        'save_json': False,         # Save results to JSON
        'conf': None,               # Confidence threshold (None = default)
        'iou': 0.7,                 # IoU threshold for NMS
        'max_det': 300,             # Max detections per image
        'half': False,            
        'dnn': False,               # Use OpenCV DNN
        'rect': False,              # Rectangular training
        'resume': RESUME_TRAINING,  # Resume from checkpoint
        'freeze': None,             # Freeze layers
        'multi_scale': True,        # ⭐ Multi-scale training (importante!)
        
        
    }
    
    # Print parametri chiave
    print("Core Settings:")
    print(f"  Epochs: {params['epochs']}")
    print(f"  Batch: {params['batch']}")
    print(f"  Image Size: {params['imgsz']}")
    print(f"  Workers: {params['workers']} (Windows compatibility)")
    print(f"  Cache: {params['cache']}")
    print(f"  Multi-scale: {params['multi_scale']}")
    
    print("\nAugmentation (Motion & Perspective):")
    print(f"  Rotation: ±{params['degrees']}°")
    print(f"  Scale: {params['scale']}")
    print(f"  Shear: {params['shear']}")
    print(f"  Perspective: {params['perspective']}")
    print(f"  Mixup: {params['mixup']}")
    print(f"  Random Erasing: {params['erasing']}")
    print(f"  Auto Augment: {params['auto_augment']}")
    
    print("\nLearning Rate:")
    print(f"  Initial: {params['lr0']}")
    print(f"  Final: {params['lrf']}")
    print(f"  Scheduler: Cosine")
    
    print("\n" + "="*60)
    print("⏱️  ESTIMATED TIME")
    print("="*60)
    print(f"  ~15-20 hours for {params['epochs']} epochs")
    print(f"  ~4.5-6 min/epoch average")
    print(f"  Checkpoint auto-saved every {params['save_period']} epochs")
    print("="*60)
    
    print("\n💡 TIPS:")
    print("  • Press Ctrl+C to pause (checkpoint saved)")
    print("  • Run script again to resume from last checkpoint")
    print("  • Monitor GPU usage: watch -n 1 nvidia-smi")
    print("  • Check results: basketball_training/yolo11s_5classes/")
    print("="*60)
    
    input("\n👉 Press ENTER to start training or Ctrl+C to cancel...")
    
    # ==================== START TRAINING ====================
    print("\n" + "🚀 "*30)
    print("TRAINING STARTED!")
    print("🚀 "*30 + "\n")
    
    try:
        results = model.train(**params)
        
        # ==================== TRAINING COMPLETATO ====================
        print("\n\n" + "🎉"*30)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉"*30)
        
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        
        # Validazione finale
        val_results = model.val()

        # Estrai metriche in modo robusto (supporta dict o oggetto)
        if isinstance(val_results, dict):
            vm_map50 = val_results.get('metrics/mAP50', val_results.get('metrics/mAP50(B)', 0))
            vm_map5095 = val_results.get('metrics/mAP50-95', val_results.get('metrics/mAP50-95(B)', 0))
            vprecision = val_results.get('metrics/precision', val_results.get('metrics/precision(B)', 0))
            vrecall = val_results.get('metrics/recall', val_results.get('metrics/recall(B)', 0))
        else:
            # prova ad accedere come oggetto
            vm_map50 = getattr(getattr(val_results, 'box', val_results), 'map50', getattr(val_results, 'map50', 0))
            vm_map5095 = getattr(getattr(val_results, 'box', val_results), 'map', getattr(val_results, 'map', 0))
            vprecision = getattr(getattr(val_results, 'box', val_results), 'mp', getattr(val_results, 'precision', 0))
            vrecall = getattr(getattr(val_results, 'box', val_results), 'mr', getattr(val_results, 'recall', 0))

        print(f"\n🎯 Performance Metrics:")
        print(f"  mAP@50:     {vm_map50:.4f} {'✅' if vm_map50 >= 0.90 else '⚠️'}")
        print(f"  mAP@50-95:  {vm_map5095:.4f}")
        print(f"  Precision:  {vprecision:.4f} {'✅' if vprecision >= 0.90 else '⚠️'}")
        print(f"  Recall:     {vrecall:.4f} {'✅' if vrecall >= 0.90 else '⚠️'}")

        # Verifica target
        if vm_map50 >= 0.90 and vprecision >= 0.90 and vrecall >= 0.90:
            print("\n✅ TARGET ACHIEVED! All metrics ≥ 90%")
        else:
            print("\n⚠️  Target not fully achieved. Consider:")
            print("  • Increase epochs")
            print("  • Fine-tune augmentation parameters")
            print("  • Check dataset quality")

        
    except KeyboardInterrupt:
        print("\n\n" + "⚠️ "*30)
        print("TRAINING INTERRUPTED BY USER")
        print("⚠️ "*30)
        print(f"\n📁 Last checkpoint saved: {RESUME_PATH}")
        print("🔄 To resume: Run this script again")
        print("="*60)
        
    except Exception as e:
        print(f"\n\n❌ ERROR DURING TRAINING:")
        print(f"   {str(e)}")
        print(f"\n📁 Check logs at: basketball_training/yolo11s_5classes/")
        raise

if __name__ == '__main__':
    print("\n")
    print("🏀 "*30)
    print(" "*20 + "BASKETBALL YOLO11 TRAINING")
    print("🏀 "*30)
    print("\n")
    
    train_model()