import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import uvicorn
import threading
from collections import deque
from enum import Enum
import time
import json  # Added to parse custom thresholds

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration for the application."""
    # Paths
    UPLOAD_DIR = Path("uploads")
    PROCESSED_DIR = Path("processed")
    MODEL_PATH = Path("basketball_training/yolo11s_5classes/weights/best.pt")

    # Video Constraints
    MAX_DURATION_SECONDS = 180  # Max processing limit (3 mins)
    TEST_MODE_DURATION = 15     # Duration for test mode

    # Retention Policy
    RETENTION_SECONDS = 1800    # 30 Minutes: Files older than this are auto-deleted
    CLEANUP_INTERVAL = 60       # Run cleanup check every 60 seconds

    # Physics & Rules (Time in seconds)
    SHOT_COOLDOWN = 1.5     # if the model recognizes a shot,  it wait 1.5 seconds before counting another. Prevente a single shot from being counted 10 times in 10 consecutive frames 
    BASKET_COOLDOWN = 2.0   # the same for the basket recognition 
    ANIMATION_DURATION = 2.0

    # Confidence Thresholds
    THRESHOLDS = {
        0: 0.6,     # Ball
        1: 0.25,    # Ball in Basket
        2: 0.7,     # Player
        3: 0.7,     # Basket
        4: 0.77     # Player Shooting
    }

    # Colors (BGR Format for OpenCV)
    COLORS = {
        0: (0, 165, 255),    # Ball (Orange)
        1: (0, 215, 255),    # Ball in Basket (Gold)
        2: (0, 255, 0),      # Player (Green)
        3: (0, 0, 255),      # Basket (Red)
        4: (255, 100, 0),    # Player Shooting (Blue)
    }
    
    CLASSES = {
        0: "Ball",
        1: "Ball in Basket",
        2: "Player",
        3: "Basket",
        4: "Player Shooting"
    }

# Ensure directories exist
Config.UPLOAD_DIR.mkdir(exist_ok=True)
Config.PROCESSED_DIR.mkdir(exist_ok=True)

# ==================== ENUMS ====================
class ProcessingMode(str, Enum):
    STATS_ONLY = "stats_only"
    STATS_EFFECTS = "stats_effects"
    FULL_TRACKING = "full_tracking"

# ==================== GLOBAL STATE ====================
processing_status = {}
stop_flags = {}

# ==================== AI MODEL ====================
def load_model():
    """Loads the YOLO model with error handling."""
    print("🔄 Loading AI Model...")
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {Config.MODEL_PATH}")
    model = YOLO(str(Config.MODEL_PATH))
    print("✅ Model loaded successfully!")
    return model

yolo_model = load_model()

# ==================== CLEANUP SERVICE ====================
class AutoCleanup:
    """Background service to delete old files and free up space."""
    
    @staticmethod
    def start():
        """Starts the cleanup thread."""
        thread = threading.Thread(target=AutoCleanup._cleanup_loop, daemon=True)
        thread.start()
        print(f"🧹 Auto-Cleanup started. Retention: {Config.RETENTION_SECONDS}s")

    @staticmethod
    def _cleanup_loop():
        """Runs periodically to remove old files."""
        while True:
            time.sleep(Config.CLEANUP_INTERVAL)
            try:
                now = time.time()
                deleted_count = 0
                
                # 1. Clean Uploads
                for f in Config.UPLOAD_DIR.iterdir():
                    if f.is_file() and (now - f.stat().st_mtime) > Config.RETENTION_SECONDS:
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception: pass

                # 2. Clean Processed Videos
                for f in Config.PROCESSED_DIR.iterdir():
                    if f.is_file() and (now - f.stat().st_mtime) > Config.RETENTION_SECONDS:
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception: pass

                # 3. Clean Memory (Status Dictionary) (To avoid Memory Leak)
                # Remove keys that haven't been updated in a while (using a simplified heuristic here)
                # Since we don't timestamp status updates, we'll just check if the file exists on disk.
                # If file is gone (deleted above), remove status.
                keys_to_remove = []
                for file_id in processing_status:
                    # Check if any file related to this ID still exists
                    has_files = any(Config.UPLOAD_DIR.glob(f"{file_id}.*")) or \
                                any(Config.PROCESSED_DIR.glob(f"{file_id}*"))
                    
                    if not has_files and processing_status[file_id]['status'] != 'processing':
                        keys_to_remove.append(file_id)
                
                for k in keys_to_remove:
                    del processing_status[k]

                if deleted_count > 0:
                    print(f"🧹 Auto-Cleanup: Removed {deleted_count} old files.")

            except Exception as e:
                print(f"⚠️ Cleanup Error: {e}")

# ==================== LOGIC CLASSES ====================

class GameStats:
    """Handles the logic for tracking shots, baskets, and percentages."""
    def __init__(self, fps):
        self.fps = fps
        self.shots_attempted = 0
        self.baskets_made = 0
        
        #calculate how many frames the cooldown lasts based on the FPS of the video
        self.shot_cooldown_frames = int(fps * Config.SHOT_COOLDOWN)
        self.basket_cooldown_frames = int(fps * Config.BASKET_COOLDOWN)
        self.anim_duration_frames = int(fps * Config.ANIMATION_DURATION)
        
        self.last_shot_frame = -self.shot_cooldown_frames
        self.last_basket_frame = -self.basket_cooldown_frames
        
        self.basket_position = None
        self.last_known_basket_pos = None
        self.animation_frames = deque(maxlen=self.anim_duration_frames)

    # called when the model recognized a player shooting 
    def register_shot(self, frame_idx):
        if frame_idx - self.last_shot_frame >= self.shot_cooldown_frames:
            self.shots_attempted += 1
            self.last_shot_frame = frame_idx
            return True
        return False

    # called when the model recognized the ball in basket 
    def register_basket(self, frame_idx, position=None):
        if frame_idx - self.last_basket_frame >= self.basket_cooldown_frames:
            # if there is a basket but there was no recent shot, it automatically adds a shot (because you can't score without shooting, so AI missed the shot)
            if (frame_idx - self.last_shot_frame) > (self.shot_cooldown_frames * 2):
                self.shots_attempted += 1
                self.last_shot_frame = frame_idx
                # print(f"   ⚠️  Basket detected without shot. Auto-added shot.")

            self.baskets_made += 1
            self.last_basket_frame = frame_idx
            self.basket_position = position
            
            self.animation_frames.clear()
            for i in range(self.anim_duration_frames):
                self.animation_frames.append(frame_idx + i)
            return True
        return False

    #calculate the shoot percentage (%)
    @property
    def accuracy(self):
        if self.shots_attempted == 0: return 0.0
        return (self.baskets_made / self.shots_attempted) * 100

    def get_animation_progress(self, current_frame):
        if current_frame not in self.animation_frames: return 0.0
        delta = current_frame - self.last_basket_frame
        return min(1.0, delta / self.anim_duration_frames)


class Visualizer:
    """Handles all drawing operations on the video frames."""
    
    # Draw animation (pulsing concentric circles) when you score a basket. (Use math,sin and alpha, to make a fade effect)
    @staticmethod
    def draw_basket_effect(frame, center_pos, progress):
        if not center_pos: return
        cx, cy = center_pos
        alpha = 1.0
        if progress < 0.15: alpha = progress / 0.15
        elif progress > 0.85: alpha = (1.0 - progress) / 0.15
        
        overlay = frame.copy()
        for i in range(4):
            delay = i * 0.1
            local_prog = max(0, min(1, (progress - delay) / (1 - delay)))
            if local_prog > 0:
                radius = int(20 + local_prog * 100)
                thickness = max(2, int(8 * (1 - local_prog)))
                cv2.circle(overlay, (cx, cy), radius, (0, 215, 255), thickness)
        
        pulse = 1.0 + np.sin(progress * np.pi * 4) * 0.3
        cv2.circle(overlay, (cx, cy), int(15 * pulse), (0, 255, 255), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.3, 0, frame)

    # Draw the scoreboard at the bottom (Shots, Baskets, Accuracy). Make it semi-transparent to make it readable.
    @staticmethod
    def draw_hud(frame, stats, w, h):
        panel_h, panel_w = 100, min(700, w - 30)
        x, y = 15, h - panel_h - 15
        
        sub_img = frame[y:y+panel_h, x:x+panel_w]
        white_rect = np.full(sub_img.shape, 30, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.2, white_rect, 0.8, 0)
        frame[y:y+panel_h, x:x+panel_w] = res
        cv2.rectangle(frame, (x, y), (x+panel_w, y+panel_h), (0, 200, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        def draw_stat_col(offset_x, label, value, color=(255,255,255)):
            cv2.putText(frame, label, (x + offset_x, y + 30), font, 0.5, (180,180,180), 1)
            cv2.putText(frame, str(value), (x + offset_x, y + 70), font, 1.3, color, 3)

        col_w = panel_w // 3
        draw_stat_col(20, "SHOTS", stats.shots_attempted)
        draw_stat_col(20 + col_w, "BASKETS", stats.baskets_made, (0, 255, 100))
        
        acc_x = x + 2 * col_w + 20
        cv2.putText(frame, "ACCURACY", (acc_x, y + 30), font, 0.5, (180,180,180), 1)
        cv2.putText(frame, f"{stats.accuracy:.1f}%", (acc_x, y + 70), font, 1.0, (0, 255, 255), 2)
        
        bar_x, bar_y = acc_x, y + 80
        bar_w = col_w - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (60,60,60), -1)
        fill_w = int((stats.accuracy / 100) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), (0, 200, 255), -1)

    # Creates the final green/gray screen with a summary of all statistics.
    @staticmethod
    def draw_final_screen(w, h, stats, total_frames, fps):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h): canvas[i, :] = [(20 + i/h*20)]*3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(canvas, "PERFORMANCE SUMMARY", (w//2 - 250, h//4), font, 1.5, (255,255,255), 3)
        data = [
            ("Total Shots", str(stats.shots_attempted)),
            ("Baskets Made", str(stats.baskets_made)),
            ("Missed Shots", str(stats.shots_attempted - stats.baskets_made)),
            ("Accuracy", f"{stats.accuracy:.1f}%"),
            ("Duration", f"{int(total_frames/fps)} sec")
        ]
        start_y = h//3
        for i, (label, val) in enumerate(data):
            y_pos = start_y + (i * 60)
            cv2.putText(canvas, label, (w//2 - 200, y_pos), font, 1.0, (200,200,200), 2)
            cv2.putText(canvas, val, (w//2 + 100, y_pos), font, 1.0, (0,255,100) if "%" in val else (255,255,255), 2)
            cv2.line(canvas, (w//2 - 200, y_pos + 20), (w//2 + 200, y_pos + 20), (50,50,50), 1)
        return canvas

class VideoProcessor:
    """Manages the video processing loop."""
    def __init__(self, file_id, input_path, output_path, test_mode, mode: ProcessingMode, thresholds: dict = None):
        self.file_id = file_id
        self.input_path = input_path
        self.output_path = output_path
        self.test_mode = test_mode
        self.mode = mode
        # Use custom thresholds if provided, otherwise use defaults
        self.thresholds = thresholds if thresholds else Config.THRESHOLDS
        
    def run(self):
        try:
            # open video 
            cap = cv2.VideoCapture(str(self.input_path))
            if not cap.isOpened(): raise RuntimeError("Could not open video file.")

            # reading video settings (FPS, width, height, total frames)    
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.test_mode: max_frames = int(fps * Config.TEST_MODE_DURATION)
            else: max_frames = min(total_frames, int(fps * Config.MAX_DURATION_SECONDS))
            
            # prepare a "blank cassette" where we'll record the edited video. It must have the same dimensions (width, height) and frame rate (fps) as the original.
            writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            # prepare the "scoreboard" by resetting it
            stats = GameStats(fps)
            frame_idx = 0
            
            self._update_status("processing", 0, max_frames, stats)
            print(f"🎬 Processing {self.file_id} | Mode: {self.mode.value} | Frames: {max_frames}")

            # read frame by frame using a while loop
            while cap.isOpened() and frame_idx < max_frames:
                # check if the user is pressed "Stop"
                if stop_flags.get(self.file_id, False):
                    print(f"🛑 Stopped by user.")
                    break

                # take the single current frame    
                success, frame = cap.read()
                if not success: break
                
                # create a copy of the original frame (it is a good practice)
                annotated = frame.copy()
                
                # --- TRACKING --- (the model analyzes the original frame, but modifies the copy )
                # detection using the model
                results = yolo_model.track(
                    frame, persist=True, verbose=False, 
                    conf=0.25, tracker="bytetrack.yaml", imgsz=640
                )
                
                # --- LOGIC --- 
                # Update scores if it finds shots or baskets. 
                self._process_detections(results, stats, frame_idx)
                
                # --- DRAWING LOGIC BASED ON MODE ---
                
                # 1. Boxes (Only in FULL_TRACKING)
                if self.mode == ProcessingMode.FULL_TRACKING:
                    self._draw_yolo_boxes(annotated, results)
                
                # 2. Effects (In FULL_TRACKING or STATS_EFFECTS)
                if self.mode in [ProcessingMode.FULL_TRACKING, ProcessingMode.STATS_EFFECTS]:
                    if stats.get_animation_progress(frame_idx) > 0:
                        Visualizer.draw_basket_effect(annotated, stats.basket_position, stats.get_animation_progress(frame_idx))
                
                # 3. HUD (Always visible in all modes)
                Visualizer.draw_hud(annotated, stats, w, h)
                
                # Writes the modified frame to the new video file.
                writer.write(annotated)
                frame_idx += 1
                
                # every 30 frames the status is updated (and the progress bar)
                if frame_idx % 30 == 0:
                    self._update_status("processing", frame_idx, max_frames, stats)

            if stop_flags.get(self.file_id, False):
                self._update_status("stopped", frame_idx, max_frames, stats)
            else:
                # create final screen and add it to the final video for 5 seconds
                summary_frame = Visualizer.draw_final_screen(w, h, stats, frame_idx, fps)
                for _ in range(int(fps * 5)): writer.write(summary_frame)
                self._update_status("completed", frame_idx, max_frames, stats)
                print(f"✅ Finished. Acc: {stats.accuracy:.1f}%")

            # close the files
            cap.release()
            writer.release()
            if self.file_id in stop_flags: del stop_flags[self.file_id]

        except Exception as e:
            print(f"❌ Error: {e}")
            processing_status[self.file_id] = {"status": "error", "message": str(e)}

    # Used to understand what is happening in the game and update the score
    def _process_detections(self, results, stats, frame_idx):
        # If the model didn't see anything in this frame (black or blank screen), exit immediately to save time.
        if not results[0].boxes: return
        # Scroll through the list of all found objects
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Use instance-specific thresholds
            if conf < self.thresholds.get(cls, 0.3): continue
            
            # The model returns a rectangle. Here we calculate the exact center point of that rectangle. It's essential to know where the ball is flying.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1+x2)//2, (y1+y2)//2)
            
            # If the model sees the basket, it stores its position (last_known_basket_pos). This is necessary because if the ball goes in, we need to know where to draw the animation.
            if cls == 3: stats.last_known_basket_pos = center
            # Player who shots
            elif cls == 4: stats.register_shot(frame_idx)
            # Ball in the basket: It also passes the position (target_pos) so the graphics system will know where to draw the visual explosion.
            elif cls == 1:
                target_pos = stats.last_known_basket_pos or center
                stats.register_basket(frame_idx, target_pos)

    # Used to show the human what the model sees. Draw the colored rectangles.
    def _draw_yolo_boxes(self, frame, results):
        if not results[0].boxes: return
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Use instance-specific thresholds
            if conf < self.thresholds.get(cls, 0.3): continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = Config.COLORS.get(cls, (255,255,255))
            label = f"{Config.CLASSES.get(cls)} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Used to update the status and the progress bar.
    def _update_status(self, status, current, total, stats):
        processing_status[self.file_id] = {
            "status": status,
            "progress": current,
            "total": total,
            "percentage": int((current/total)*100) if total > 0 else 0,
            "stats": {"shots": stats.shots_attempted, "baskets": stats.baskets_made, "accuracy": stats.accuracy}
        }

# ==================== FASTAPI APP ====================
app = FastAPI(title="Basketball AI Tracker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.on_event("startup")
def startup_event():
    """Starts background services on app startup."""
    AutoCleanup.start()

@app.get("/")
def home(): return {"message": "Basketball AI Tracker is Running", "docs": "/docs"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"): raise HTTPException(400, "File must be a video.")
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = Config.UPLOAD_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f: shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "filename": file.filename}

@app.post("/process/{file_id}")
async def start_process(file_id: str, test_mode: bool = False, mode: ProcessingMode = ProcessingMode.FULL_TRACKING, thresholds: str = None):
    input_files = list(Config.UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files: raise HTTPException(404, "Video not found.")
    
    # Parse custom thresholds if provided
    active_thresholds = Config.THRESHOLDS.copy()
    if thresholds:
        try:
            custom_vals = json.loads(thresholds)
            # Convert string keys to integers
            for k, v in custom_vals.items():
                active_thresholds[int(k)] = float(v)
            print(f"⚙️ Using custom thresholds for {file_id}: {active_thresholds}")
        except Exception as e:
            print(f"⚠️ Failed to parse thresholds: {e}. Using defaults.")

    processor = VideoProcessor(file_id, input_files[0], Config.PROCESSED_DIR / f"{file_id}_processed.mp4", test_mode, mode, active_thresholds)
    thread = threading.Thread(target=processor.run)
    thread.start()
    return {"status": "started", "file_id": file_id, "mode": mode}

@app.get("/status/{file_id}")
def get_status(file_id: str): return processing_status.get(file_id, {"status": "not_found"})

@app.post("/stop/{file_id}")
def stop_process(file_id: str):
    if file_id in processing_status and processing_status[file_id]['status'] == 'processing':
        stop_flags[file_id] = True
        return {"message": "Stopping..."}
    return {"message": "Not processing or not found"}

@app.get("/download/{file_id}")
def download_result(file_id: str):
    path = Config.PROCESSED_DIR / f"{file_id}_processed.mp4"
    if not path.exists(): raise HTTPException(404, "File not ready.")
    
    # We don't delete immediately here to allow retries. 
    # The AutoCleanup service will handle it after RETENTION_SECONDS.
    return FileResponse(path, media_type="video/mp4", filename=f"basket_ai_{file_id}.mp4")

if __name__ == "__main__":
    print("\n🏀 SERVER STARTING...")
    uvicorn.run(app, host="0.0.0.0", port=8000)