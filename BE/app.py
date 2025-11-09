"""
🏀 BASKETBALL TRACKER API with STATS
Tracks 5 classes with different colors and calculates shooting statistics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import shutil
import uuid
import uvicorn
import threading
import os
from dotenv import load_dotenv
from collections import deque

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

processing_status = {}
stop_flags = {}

# ==================== CARICA MODELLO ====================
print("🔄 Loading custom trained model...")

CUSTOM_MODEL_PATH = Path("basketball_training/yolo11s_5classes/weights/best.pt")
if not CUSTOM_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ Trained model not found at {CUSTOM_MODEL_PATH}\n"
        f"   Please train the model first using train_basketball_model.py"
    )

yolo_model = YOLO(str(CUSTOM_MODEL_PATH))

print("✅ Model loaded successfully!")
print(f"📋 Classes: {yolo_model.names}")

# ==================== CONFIGURAZIONE COLORI ====================
CLASS_COLORS = {
    0: (0, 165, 255),    # ball - Orange
    1: (0, 215, 255),    # ball-in-basket - Gold/Yellow
    2: (0, 255, 0),      # player - Green
    3: (0, 0, 255),      # basket - Red
    4: (255, 100, 0),    # player-shooting - Blue
}

CLASS_NAMES = {
    0: "Ball",
    1: "Ball in Basket",
    2: "Player",
    3: "Basket",
    4: "Player Shooting"
}

CLASS_THICKNESS = {
    0: 3,
    1: 4,
    2: 2,
    3: 2,
    4: 3,
}

# ==================== CONFIGURAZIONE STATISTICHE (IN SECONDI) ====================
SHOT_COOLDOWN_SECONDS = 1.5      # Cooldown tiri (secondi)
BASKET_COOLDOWN_SECONDS = 2.0    # Cooldown canestri (secondi)
BASKET_ANIMATION_SECONDS = 2.0   # Durata animazione (secondi)

class BasketballStats:
    """Classe per gestire le statistiche di gioco"""
    def __init__(self, fps):
        self.fps = fps
        # Converti secondi in frame in base al framerate
        self.shot_cooldown_frames = int(fps * SHOT_COOLDOWN_SECONDS)
        self.basket_cooldown_frames = int(fps * BASKET_COOLDOWN_SECONDS)
        self.basket_animation_frames_duration = int(fps * BASKET_ANIMATION_SECONDS)
        
        self.shots_attempted = 0
        self.baskets_made = 0
        self.last_shot_frame = -self.shot_cooldown_frames
        self.last_basket_frame = -self.basket_cooldown_frames
        self.basket_animation_frames = deque(maxlen=self.basket_animation_frames_duration)

        #Salva posizione canestro per animazione
        self.basket_position = None  # (x, y) del centro del canestro
        self.last_known_basket_position = None  #ultima posizione nota
        
        print(f"📊 Stats Config (FPS={fps}):")
        print(f"   Shot cooldown: {SHOT_COOLDOWN_SECONDS}s = {self.shot_cooldown_frames} frames")
        print(f"   Basket cooldown: {BASKET_COOLDOWN_SECONDS}s = {self.basket_cooldown_frames} frames")
        print(f"   Animation duration: {BASKET_ANIMATION_SECONDS}s = {self.basket_animation_frames_duration} frames")
        
    def register_shot(self, current_frame):
        """Registra un tiro se è passato abbastanza tempo"""
        if current_frame - self.last_shot_frame >= self.shot_cooldown_frames:
            self.shots_attempted += 1
            self.last_shot_frame = current_frame
            return True
        return False
    
    def register_basket(self, current_frame, basket_position=None):
        """Registra un canestro se è passato abbastanza tempo"""
        if current_frame - self.last_basket_frame >= self.basket_cooldown_frames:
            # FIX: Se NON c'è stato un tiro RECENTE (entro una finestra temporale ragionevole), conta anche il tiro
            frames_since_last_shot = current_frame - self.last_shot_frame
            
            # Se l'ultimo tiro è stato rilevato MOLTO tempo fa (oltre il doppio del cooldown)
            # significa che il tiro attuale non è stato rilevato
            if frames_since_last_shot > (self.shot_cooldown_frames * 2):
                self.shots_attempted += 1
                self.last_shot_frame = current_frame
                print(f"   ⚠️  Canestro senza tiro rilevato - aggiunto tiro automatico")
            
            self.baskets_made += 1
            self.last_basket_frame = current_frame

            # Salva posizione del canestro
            self.basket_position = basket_position

            # Attiva animazione
            self.basket_animation_frames = deque(maxlen=self.basket_animation_frames_duration)
            for i in range(self.basket_animation_frames_duration):
                self.basket_animation_frames.append(current_frame + i)
            return True
        return False
    
    def get_percentage(self):
        """Calcola la percentuale di tiro"""
        if self.shots_attempted == 0:
            return 0.0
        return (self.baskets_made / self.shots_attempted) * 100
    
    def is_animating(self, current_frame):
        """Controlla se l'animazione canestro è attiva"""
        return current_frame in self.basket_animation_frames
    
    def get_animation_progress(self, current_frame):
        """Ottieni il progresso dell'animazione (0.0 a 1.0)"""
        if not self.is_animating(current_frame):
            return 0.0
        frames_since_basket = current_frame - self.last_basket_frame
        return min(1.0, frames_since_basket / self.basket_animation_frames_duration)

# ==================== CONFIGURAZIONE ====================
MAX_DURATION = 180  # 3 minuti max di PROCESSING (anche se il video è più lungo)
TEST_MODE_DURATION = 15

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "message": "Basketball Tracker API with Stats", 
        "status": "running",
        "model": str(CUSTOM_MODEL_PATH),
        "classes": CLASS_NAMES
    }

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")
    
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    input_path = UPLOAD_DIR / f"{file_id}{ext}"
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # Non bloccare upload di video lunghi, ma avvisa che verrà processato solo fino a 3 min
    will_be_truncated = duration > MAX_DURATION
    
    return {
        "file_id": file_id, 
        "filename": file.filename, 
        "duration": duration,
        "fps": fps,
        "frames": int(frame_count),
        "will_be_truncated": will_be_truncated,
        "max_processing_duration": MAX_DURATION if will_be_truncated else None,
        "warning": f"Video will be processed up to {MAX_DURATION}s (first 3 minutes)" if will_be_truncated else None
    }

@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get processing status"""
    if file_id not in processing_status:
        return {"status": "not_found", "progress": 0, "total": 0}
    return processing_status[file_id]

def draw_basket_animation(frame, stats, progress):
    """Disegna animazione semplice del canestro con cerchi concentrici"""
    if stats.basket_position is None:
        return  # Nessuna posizione salvata, non disegna nulla
    
    center_x, center_y = stats.basket_position
    
    # Effetto di fade in/out smooth
    if progress < 0.15:
        alpha = progress / 0.15
    elif progress > 0.85:
        alpha = (1.0 - progress) / 0.15
    else:
        alpha = 1.0
    
    overlay = frame.copy()
    
    # Cerchi concentrici che si espandono
    num_rings = 4
    for i in range(num_rings):
        delay = i * 0.1  # Ritardo tra un cerchio e l'altro
        local_progress = max(0, min(1, (progress - delay) / (1 - delay)))
        
        if local_progress > 0:
            # Raggio che cresce
            max_radius = 100  # Raggio massimo più piccolo
            radius = int(20 + local_progress * max_radius)
            
            # Spessore che diminuisce
            thickness = max(2, int(8 * (1 - local_progress)))
            
            # Trasparenza che diminuisce
            ring_alpha = (1 - local_progress) * alpha * 0.8
            
            # Disegna cerchio giallo/oro
            color = (0, 215, 255)  # Gold/Yellow
            cv2.circle(overlay, (center_x, center_y), radius, color, thickness)
    
    # Piccolo cerchio centrale che pulsa
    pulse_scale = 1.0 + np.sin(progress * np.pi * 4) * 0.3
    inner_radius = int(15 * pulse_scale)
    cv2.circle(overlay, (center_x, center_y), inner_radius, (0, 255, 255), -1)
    cv2.circle(overlay, (center_x, center_y), inner_radius, (255, 255, 255), 2)
    
    # Blend con alpha
    cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.3, 0, frame)

def draw_stats_overlay(frame, stats, width, height):
    """Disegna overlay statistiche moderno e professionale"""
    # Dimensioni overlay più largo
    overlay_height = 100
    overlay_y = height - overlay_height - 15
    overlay_x = 15
    overlay_width = min(700, width - 30)  # Più largo, responsive
    
    # Background moderno con gradiente
    overlay_bg = frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width].copy()
    
    # Gradiente scuro
    gradient = np.zeros_like(overlay_bg, dtype=np.uint8)
    for i in range(overlay_height):
        intensity = int(15 + (i / overlay_height) * 10)
        gradient[i, :] = [intensity, intensity, intensity]
    
    # Blend gradiente
    overlay_bg = cv2.addWeighted(overlay_bg, 0.2, gradient, 0.8, 0)
    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = overlay_bg
    
    # Bordo moderno con accent color
    border_color = (0, 200, 255)
    cv2.rectangle(frame, 
                  (overlay_x, overlay_y), 
                  (overlay_x + overlay_width, overlay_y + overlay_height),
                  border_color, 2)
    
    # Linea accent in alto
    cv2.line(frame,
             (overlay_x, overlay_y),
             (overlay_x + overlay_width, overlay_y),
             (0, 255, 255), 4)
    
    # Layout a 3 colonne
    col_width = overlay_width // 3
    padding = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # COLONNA 1: Tiri
    x1 = overlay_x + padding
    y_label = overlay_y + 30
    y_value = overlay_y + 65
    
    cv2.putText(frame, "TIRI", (x1, y_label),
                font, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, str(stats.shots_attempted), (x1, y_value),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    
    # COLONNA 2: Canestri
    x2 = overlay_x + col_width + padding
    cv2.putText(frame, "CANESTRI", (x2, y_label),
                font, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, str(stats.baskets_made), (x2, y_value),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3)
    
    # COLONNA 3: Percentuale con barra
    x3 = overlay_x + 2 * col_width + padding
    percentage = stats.get_percentage()
    
    cv2.putText(frame, "PRECISIONE", (x3, y_label),
                font, 0.5, (180, 180, 180), 1)
    
    # Percentuale numero
    perc_text = f"{percentage:.1f}%"
    cv2.putText(frame, perc_text, (x3, y_value),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Barra progressiva moderna sotto la percentuale
    bar_x = x3
    bar_y = y_value + 10
    bar_width = col_width - padding - 10
    bar_height = 8
    
    # Background barra (scuro)
    cv2.rectangle(frame, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height),
                  (40, 40, 40), -1)
    
    # Bordo barra
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (80, 80, 80), 1)
    
    # Barra riempita con gradiente
    fill_width = int((percentage / 100) * bar_width)
    if fill_width > 2:
        # Colore dinamico
        if percentage >= 60:
            color = (0, 255, 100)  # Verde
        elif percentage >= 40:
            color = (0, 200, 255)  # Giallo/Gold
        else:
            color = (0, 100, 255)  # Arancione
        
        # Riempimento con leggero gradiente
        for i in range(fill_width):
            intensity = 0.7 + (i / fill_width) * 0.3
            c = tuple(int(x * intensity) for x in color)
            cv2.line(frame, (bar_x + i, bar_y + 1),
                    (bar_x + i, bar_y + bar_height - 1), c, 1)
        
        # Highlight sulla barra
        cv2.line(frame, (bar_x, bar_y + 1),
                (bar_x + fill_width, bar_y + 1),
                (255, 255, 255), 1)

def draw_final_stats_screen(width, height, stats, total_frames, fps):
    """Crea schermata finale moderna e professionale"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradiente moderno (scuro elegante)
    for i in range(height):
        intensity_r = int(10 + (i / height) * 15)
        intensity_g = int(12 + (i / height) * 18)
        intensity_b = int(15 + (i / height) * 20)
        frame[i, :] = [intensity_b, intensity_g, intensity_r]
    
    # Overlay scuro centrale per contrast
    overlay_margin = 100
    cv2.rectangle(frame,
                  (overlay_margin, overlay_margin),
                  (width - overlay_margin, height - overlay_margin),
                  (0, 0, 0), -1)
    
    # Blend overlay
    overlay_alpha = 0.6
    overlay_region = frame[overlay_margin:height-overlay_margin, overlay_margin:width-overlay_margin]
    frame[overlay_margin:height-overlay_margin, overlay_margin:width-overlay_margin] = \
        cv2.addWeighted(overlay_region, 1-overlay_alpha, 
                       np.zeros_like(overlay_region), overlay_alpha, 0)
    
    # Bordo accent moderno
    cv2.rectangle(frame,
                  (overlay_margin, overlay_margin),
                  (width - overlay_margin, height - overlay_margin),
                  (0, 200, 255), 3)
    
    # Linea accent superiore
    cv2.line(frame,
             (overlay_margin, overlay_margin),
             (width - overlay_margin, overlay_margin),
             (0, 255, 255), 6)
    
    # Titolo principale moderno
    title = "STATISTICHE FINALI"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 2.2
    title_thickness = 4
    
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
    title_x = width // 2 - tw // 2
    title_y = overlay_margin + 80
    
    # Shadow
    cv2.putText(frame, title, (title_x + 2, title_y + 2),
                font, title_scale, (0, 0, 0), title_thickness + 2)
    
    # Testo principale
    cv2.putText(frame, title, (title_x, title_y),
                font, title_scale, (255, 255, 255), title_thickness)
    
    # Accent line sotto titolo
    line_y = title_y + 20
    line_width = 300
    cv2.line(frame,
             (width // 2 - line_width // 2, line_y),
             (width // 2 + line_width // 2, line_y),
             (0, 255, 255), 3)
    
    # Container statistiche
    stats_top = line_y + 60
    stats_height = 320
    stats_width = 700
    stats_x = width // 2 - stats_width // 2
    
    # Background container
    cv2.rectangle(frame,
                  (stats_x, stats_top),
                  (stats_x + stats_width, stats_top + stats_height),
                  (30, 30, 30), -1)
    
    # Bordo container
    cv2.rectangle(frame,
                  (stats_x, stats_top),
                  (stats_x + stats_width, stats_top + stats_height),
                  (0, 200, 255), 2)
    
    # Statistiche in grid 2x2
    cell_height = stats_height // 2
    cell_width = stats_width // 2
    
    percentage = stats.get_percentage()
    missed = max(0, stats.shots_attempted - stats.baskets_made)
    
    stats_data = [
        ("TIRI TENTATI", str(stats.shots_attempted), (255, 255, 255), 0, 0),
        ("CANESTRI", str(stats.baskets_made), (0, 255, 100), 1, 0),
        ("TIRI SBAGLIATI", str(missed), (100, 150, 255), 0, 1),
        ("PRECISIONE", f"{percentage:.1f}%", (0, 255, 255), 1, 1)
    ]
    
    for label, value, color, col, row in stats_data:
        cell_x = stats_x + col * cell_width
        cell_y = stats_top + row * cell_height
        
        # Separatori
        if col == 1:
            cv2.line(frame, (cell_x, cell_y), (cell_x, cell_y + cell_height),
                    (60, 60, 60), 2)
        if row == 1:
            cv2.line(frame, (stats_x, cell_y), (stats_x + stats_width, cell_y),
                    (60, 60, 60), 2)
        
        # Label (piccolo, sopra)
        label_y = cell_y + 50
        cv2.putText(frame, label,
                    (cell_x + 40, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
        
        # Valore (grande, sotto)
        value_scale = 2.5
        value_thickness = 5
        (vw, vh), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, value_scale, value_thickness)
        value_x = cell_x + cell_width // 2 - vw // 2
        value_y = cell_y + cell_height // 2 + 30
        
        # Glow effect
        for i in range(3):
            cv2.putText(frame, value, (value_x, value_y),
                       cv2.FONT_HERSHEY_SIMPLEX, value_scale, color, value_thickness + 4 - i)
        
        # Valore principale
        cv2.putText(frame, value, (value_x, value_y),
                    cv2.FONT_HERSHEY_SIMPLEX, value_scale, color, value_thickness)
    
    # Footer con durata
    duration = total_frames / fps
    footer_y = stats_top + stats_height + 50
    footer_text = f"Durata analisi: {int(duration // 60)}:{int(duration % 60):02d}"
    
    (fw, fh), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, footer_text,
                (width // 2 - fw // 2, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
    
    return frame

def draw_detection(frame, box, cls, conf, track_id=None):
    """Disegna una detection sul frame con stile personalizzato"""
    x1, y1, x2, y2 = map(int, box)
    
    color = CLASS_COLORS.get(cls, (255, 255, 255))
    thickness = CLASS_THICKNESS.get(cls, 2)
    class_name = CLASS_NAMES.get(cls, f"Class_{cls}")
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    label = f"{class_name} {conf:.2f}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    cv2.rectangle(frame, 
                  (x1, y1 - label_h - baseline - 5), 
                  (x1 + label_w + 5, y1), 
                  color, -1)
    
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    if track_id is not None:
        track_label = f"ID:{track_id}"
        cv2.putText(frame, track_label, (x1, y2 + 20), 
                    font, 0.5, color, 2)

def process_video_thread(file_id: str, input_path: Path, output_path: Path, test_mode: bool):
    """Thread per processare il video"""
    try:
        cap = cv2.VideoCapture(str(input_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcola max_frames: limita a 3 minuti se non in test mode
        if test_mode:
            max_frames = int(fps * TEST_MODE_DURATION)
        else:
            max_frames = min(total_frames, int(fps * MAX_DURATION))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Inizializza statistiche con FPS dinamico
        stats = BasketballStats(fps)
        
        frame_count = 0
        processing_status[file_id] = {
            "status": "processing", 
            "progress": 0, 
            "total": max_frames,
            "detections": {name: 0 for name in CLASS_NAMES.values()},
            "stats": {
                "shots": 0,
                "baskets": 0,
                "percentage": 0.0
            }
        }
        
        stop_flags[file_id] = False

        print(f"\n{'='*60}")
        print(f"🎬 Processing video {file_id}")
        print(f"   Mode: {'TEST (15s)' if test_mode else f'FULL (max {MAX_DURATION}s)'}")
        print(f"   Total video frames: {total_frames}")
        print(f"   Processing frames: {max_frames}")
        print(f"   FPS: {fps}")
        if max_frames < total_frames:
            print(f"   ⚠️  Video truncated to first {MAX_DURATION}s")
        print(f"{'='*60}\n")
        
        while cap.isOpened() and frame_count < max_frames:
            # Controlla se è stato richiesto lo stop
            if stop_flags.get(file_id, False):
                print(f"⚠️  Processing stopped by user at frame {frame_count}/{max_frames}")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = frame.copy()
            
            # Detection + Tracking
            results = yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.3,
                iou=0.5,
                tracker="bytetrack.yaml",
                imgsz=640
            )
            
            #traccia posizione canestro
            basket_center_position = None  

            # Processa detections
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    min_conf = {
                        0: 0.6,
                        1: 0.25,
                        2: 0.7,
                        3: 0.7,
                        4: 0.77,
                    }
                    
                    if conf < min_conf.get(cls, 0.3):
                        continue
                    
                    xyxy = box.xyxy[0]
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    # Se rilevi il canestro (classe 3), salva la sua posizione centrale
                    if cls == 3:  # basket
                        x1, y1, x2, y2 = map(int, xyxy)
                        basket_center_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                        stats.last_known_basket_position = basket_center_position  # aggiorna ultima posizione nota
                    
                    if cls == 4:  # player-shooting
                        if stats.register_shot(frame_count):
                            print(f"🏀 SHOT #{stats.shots_attempted} at frame {frame_count}")
                    
                    if cls == 1:  # ball-in-basket
                        #usa posizione corrente o ultima nota
                        position_to_use = basket_center_position if basket_center_position else stats.last_known_basket_position
                        if stats.register_basket(frame_count, position_to_use):
                            print(f"🎯 BASKET #{stats.baskets_made} at frame {frame_count} - {stats.get_percentage():.1f}%")
                    
                    draw_detection(annotated, xyxy, cls, conf, track_id)
                    
                    class_name = CLASS_NAMES.get(cls, f"Class_{cls}")
                    processing_status[file_id]["detections"][class_name] += 1
            
            # Disegna statistiche real-time
            draw_stats_overlay(annotated, stats, width, height)
            
            # Animazione canestro
            if stats.is_animating(frame_count):
                progress = stats.get_animation_progress(frame_count)
                draw_basket_animation(annotated, stats, progress)  
            
            # Info frame
            info_text = f"Frame: {frame_count+1}/{max_frames}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Legenda
            legend_y = 30
            for cls, name in CLASS_NAMES.items():
                color = CLASS_COLORS[cls]
                text = f"{name}"
                cv2.putText(annotated, text, (width - 200, legend_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                legend_y += 25
            
            out.write(annotated)
            frame_count += 1
            
            # Aggiorna status
            processing_status[file_id].update({
                "status": "processing",
                "progress": frame_count,
                "total": max_frames,
                "percentage": int((frame_count / max_frames) * 100),
                "stats": {
                    "shots": stats.shots_attempted,
                    "baskets": stats.baskets_made,
                    "percentage": stats.get_percentage()
                }
            })
            
            if frame_count % 30 == 0:
                pct = processing_status[file_id]['percentage']
                print(f"⏳ Progress: {frame_count}/{max_frames} frames ({pct}%) | Shots: {stats.shots_attempted} | Baskets: {stats.baskets_made}")
        
        # Gestisci stop vs completamento normale
        was_stopped = stop_flags.get(file_id, False)
        
        if was_stopped:
            # Processing interrotto dall'utente
            cap.release()
            out.release()
            
            # Cleanup file temporaneo
            if output_path.exists():
                output_path.unlink()
            
            # Aggiorna status
            processing_status[file_id] = {
                "status": "stopped",
                "progress": frame_count,
                "total": max_frames,
                "percentage": int((frame_count / max_frames) * 100),
                "stats": {
                    "shots": stats.shots_attempted,
                    "baskets": stats.baskets_made,
                    "percentage": stats.get_percentage()
                },
                "message": f"Processing stopped at frame {frame_count}/{max_frames}"
            }
            
            # Cleanup flag
            if file_id in stop_flags:
                del stop_flags[file_id]
            
            print(f"\n{'='*60}")
            print(f"⚠️  Video {file_id} processing STOPPED by user")
            print(f"   Frames processed: {frame_count}/{max_frames}")
            print(f"{'='*60}\n")
            return

        # Schermata finale (5 secondi)
        final_screen = draw_final_stats_screen(width, height, stats, frame_count, fps)
        final_frames = fps * 5  # 5 secondi
        for _ in range(final_frames):
            out.write(final_screen)
        
        cap.release()
        out.release()
        
        # Cleanup flag
        if file_id in stop_flags:
            del stop_flags[file_id]

        processing_status[file_id].update({
            "status": "completed",
            "progress": frame_count,
            "total": max_frames,
            "percentage": 100,
            "stats": {
                "shots": stats.shots_attempted,
                "baskets": stats.baskets_made,
                "percentage": stats.get_percentage()
            }
        })
        
        print(f"\n{'='*60}")
        print(f"✅ Video {file_id} processing completed!")
        print(f"   Frames processed: {frame_count}")
        print(f"   📊 FINAL STATS:")
        print(f"      Shots Attempted: {stats.shots_attempted}")
        print(f"      Baskets Made: {stats.baskets_made}")
        print(f"      Shooting %: {stats.get_percentage():.1f}%")
        print(f"{'='*60}\n")
        
    except Exception as e:
        processing_status[file_id] = {
            "status": "error", 
            "message": str(e)
        }
        print(f"❌ Error processing video {file_id}: {e}")

@app.post("/process/{file_id}")
async def process_video(file_id: str, test_mode: bool = False):
    """Start video processing"""
    input_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files:
        raise HTTPException(404, "Video not found")
    
    input_path = input_files[0]
    output_path = PROCESSED_DIR / f"{file_id}_processed.mp4"
    
    thread = threading.Thread(
        target=process_video_thread, 
        args=(file_id, input_path, output_path, test_mode)
    )
    thread.start()
    
    return {
        "file_id": file_id, 
        "status": "started",
        "test_mode": test_mode
    }

@app.post("/stop/{file_id}")
async def stop_processing(file_id: str):
    """Stop video processing"""
    if file_id not in processing_status:
        raise HTTPException(404, "Video processing not found")
    
    current_status = processing_status[file_id].get("status")
    
    if current_status != "processing":
        raise HTTPException(400, f"Cannot stop: video is {current_status}")
    
    # Imposta flag di stop
    stop_flags[file_id] = True
    
    print(f"🛑 Stop requested for video {file_id}")
    
    return {
        "file_id": file_id,
        "status": "stop_requested",
        "message": "Processing will stop at next frame"
    }


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    """Download processed video"""
    output_path = PROCESSED_DIR / f"{file_id}_processed.mp4"
    if not output_path.exists():
        raise HTTPException(404, "Processed video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"basketball_tracked_{file_id}.mp4"
    )

@app.delete("/clear/{file_id}")
async def clear_video(file_id: str):
    """Clear uploaded and processed videos"""
    deleted_count = 0
    for path in UPLOAD_DIR.glob(f"{file_id}.*"):
        path.unlink()
        deleted_count += 1
    for path in PROCESSED_DIR.glob(f"{file_id}*"):
        path.unlink()
        deleted_count += 1
    if file_id in processing_status:
        del processing_status[file_id]
    
    # NUOVO: Cleanup flag di stop
    if file_id in stop_flags:
        del stop_flags[file_id]
    
    return {"status": "cleared", "files_deleted": deleted_count}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": yolo_model is not None,
        "model_path": str(CUSTOM_MODEL_PATH),
        "classes": CLASS_NAMES,
        "uploads_dir": str(UPLOAD_DIR.absolute()),
        "processed_dir": str(PROCESSED_DIR.absolute())
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_path": str(CUSTOM_MODEL_PATH),
        "classes": CLASS_NAMES,
        "colors": {name: f"BGR{tuple(CLASS_COLORS[idx])}" for idx, name in CLASS_NAMES.items()},
        "num_classes": len(CLASS_NAMES)
    }

if __name__ == "__main__":
    print("\n" + "🏀 "*30)
    print("BASKETBALL TRACKER API with STATS - Starting...")
    print("🏀 "*30)
    print(f"\n📁 Uploads directory: {UPLOAD_DIR.absolute()}")
    print(f"📁 Processed directory: {PROCESSED_DIR.absolute()}")
    print(f"🎯 Model: {CUSTOM_MODEL_PATH.absolute()}")
    print(f"📋 Classes: {list(CLASS_NAMES.values())}")
    print(f"\n⚙️  Timing Configuration:")
    print(f"   Shot cooldown: {SHOT_COOLDOWN_SECONDS}s")
    print(f"   Basket cooldown: {BASKET_COOLDOWN_SECONDS}s")
    print(f"   Animation duration: {BASKET_ANIMATION_SECONDS}s")
    print(f"   Max processing: {MAX_DURATION}s (first 3 minutes)")
    print(f"\n🌐 Server: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")