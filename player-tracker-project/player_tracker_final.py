import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import warnings
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚽ Player Tracker",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {padding: 0rem 0rem;}
    </style>
""", unsafe_allow_html=True)


# =====================================================
# PLAYER TRACKER CLASS
# =====================================================
class PlayerTracker:
    
    def __init__(self):
        try:
            # Load YOLO model (cached after first run)
            st.status("Loading YOLO model...", state="running")
            self.model = YOLO('yolov8m.pt')
            st.status("YOLO loaded", state="complete")
        except Exception as e:
            st.error(f"YOLO Error: {e}")
            self.model = None
        
        try:
            # Initialize DeepSort tracker
            st.status("Loading DeepSort tracker...", state="running")
            self.deepsort = DeepSort(
                embedder="mobilenet",
                max_age=30,
                n_init=2,
                nn_budget=100
            )
            st.status("DeepSort loaded", state="complete")
        except Exception as e:
            st.error(f"DeepSort Error: {e}")
            self.deepsort = None
        
        try:
            # Initialize OCR
            st.status("Loading OCR engine...", state="running")
            self.ocr = easyocr.Reader(['en'], gpu=False)
            st.status("OCR loaded", state="complete")
        except Exception as e:
            st.error(f"OCR Error: {e}")
            self.ocr = None
        
        # Tracking data
        self.tracking_data = defaultdict(
            lambda: {
                "appearances": 0,
                "shirt_numbers": [],
                "team": None,
                "positions": [],
                "frames_seen": 0
            }
        )
    
    def detect_team(self, cropped_player):
        """Detect team by jersey color"""
        
        if cropped_player is None or cropped_player.size == 0:
            return "Unknown"
        
        try:
            # Frame is BGR from OpenCV
            hsv = cv2.cvtColor(cropped_player, cv2.COLOR_BGR2HSV)
            
            # Color ranges in HSV (BGR)
            color_ranges = {
                "Red": (
                    np.array([0, 80, 80]),
                    np.array([10, 255, 255])
                ),
                "Blue": (
                    np.array([100, 80, 80]),
                    np.array([130, 255, 255])
                ),
                "White": (
                    np.array([0, 0, 150]),
                    np.array([180, 80, 255])
                ),
                "Black": (
                    np.array([0, 0, 0]),
                    np.array([180, 255, 80])
                ),
                "Yellow": (
                    np.array([15, 80, 80]),
                    np.array([35, 255, 255])
                ),
                "Green": (
                    np.array([50, 80, 80]),
                    np.array([90, 255, 255])
                )
            }
            
            max_pixels = 0
            detected_team = "Unknown"
            
            for team, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, lower, upper)
                pixels = np.count_nonzero(mask)
                
                if pixels > max_pixels:
                    max_pixels = pixels
                    detected_team = team
            
            return detected_team
        
        except Exception as e:
            return "Unknown"
    
    def extract_shirt_number(self, cropped_player):
        """Extract shirt number using OCR"""
        
        if self.ocr is None:
            return None
        
        try:
            if cropped_player is None or cropped_player.size == 0:
                return None
            
            h, w = cropped_player.shape[:2]
            
            # Focus on center region
            center_region = cropped_player[
                h // 4:3 * h // 4,
                w // 4:3 * w // 4
            ]
            
            if center_region.size == 0:
                return None
            
            # Read text
            results = self.ocr.readtext(center_region)
            
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                
                # Extract only numbers
                numbers = ''.join(c for c in text if c.isdigit())
                
                if numbers:
                    try:
                        num = int(numbers)
                        if 0 <= num <= 99 and confidence > 0.25:
                            return str(num).zfill(2)
                    except:
                        continue
            
            return None
        
        except Exception as e:
            return None
    
    def process_frame(self, frame):
        """Detect and track players"""
        
        if self.model is None or self.deepsort is None:
            return frame.copy(), {}
        
        try:
            # YOLO detection (class 0 = person)
            results = self.model(
                frame,
                classes=[0],
                conf=0.4,
                verbose=False
            )
            
            boxes = results[0].boxes
            
            if boxes is None or len(boxes) == 0:
                return frame.copy(), {}
            
            detections = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            
            # Prepare for DeepSort
            detection_list = []
            
            for box, conf in zip(detections, confidences):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                detection_list.append(
                    ([float(x1), float(y1), float(w), float(h)], float(conf), "player")
                )
            
            # DeepSort tracking
            tracks = self.deepsort.update_tracks(detection_list, frame=frame)
            
            annotated_frame = frame.copy()
            detected_players = {}
            
            # Process each track
            for track in tracks:
                
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_tlbr()
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Clamp to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop player region
                player_crop = frame[y1:y2, x1:x2]
                
                if player_crop.size == 0:
                    continue
                
                # Detect team
                team = self.detect_team(player_crop)
                
                # Extract shirt number
                shirt_num = self.extract_shirt_number(player_crop)
                
                # ========== UPDATE DATA ==========
                self.tracking_data[track_id]['appearances'] += 1
                self.tracking_data[track_id]['frames_seen'] += 1
                self.tracking_data[track_id]['positions'].append((x1, y1, x2, y2))
                
                if team != "Unknown":
                    self.tracking_data[track_id]['team'] = team
                
                if shirt_num and shirt_num not in self.tracking_data[track_id]['shirt_numbers']:
                    self.tracking_data[track_id]['shirt_numbers'].append(shirt_num)
                
                detected_players[track_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'team': team,
                    'shirt_number': shirt_num or 'N/A'
                }
                
                # ========== DRAW ==========
                color_map = {
                    'Red': (0, 0, 255),
                    'Blue': (255, 0, 0),
                    'White': (200, 200, 200),
                    'Yellow': (0, 255, 255),
                    'Black': (50, 50, 50),
                    'Green': (0, 255, 0)
                }
                
                color = color_map.get(team, (0, 255, 0))
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"ID:{track_id} | #{shirt_num or '?'} | {team}"
                
                text_y = max(25, y1 - 10)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            return annotated_frame, detected_players
        
        except Exception as e:
            st.warning(f"Frame processing error: {str(e)[:50]}")
            return frame.copy(), {}
    
    def get_stats(self):
        """Generate statistics"""
        
        stats = []
        
        for player_id, data in self.tracking_data.items():
            
            shirt_num = data['shirt_numbers'][0] if data['shirt_numbers'] else 'N/A'
            
            detection_rate = (
                data['appearances'] / max(1, data['frames_seen']) * 100
            )
            
            stats.append({
                'Player ID': player_id,
                'Shirt Number': shirt_num,
                'Team': data['team'] or 'Unknown',
                'Frames Detected': data['appearances'],
                'Detection Rate': f"{detection_rate:.1f}%"
            })
        
        if not stats:
            return pd.DataFrame()
        
        return pd.DataFrame(stats).sort_values('Frames Detected', ascending=False)


# =====================================================
# CACHE
# =====================================================
@st.cache_resource
def load_tracker():
    return PlayerTracker()


# =====================================================
# MAIN APP
# =====================================================
def main():
    
    st.title("⚽ Player Tracking System")
    st.write("Upload → Track → Analyze")
    
    # Load tracker
    tracker = load_tracker()
    
    # Check models loaded
    if tracker.model is None or tracker.deepsort is None:
        st.error("❌ Failed to load models. Check dependencies.")
        return
    
    # Upload section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_file = st.file_uploader(
            "Upload video",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv']
        )
    
    with col2:
        process_btn = st.button("Process", use_container_width=True)
    
    # =====================================================
    # PROCESSING
    # =====================================================
    if video_file is not None and process_btn:
        
        # Reset tracking
        tracker.tracking_data = defaultdict(
            lambda: {
                'appearances': 0,
                'shirt_numbers': [],
                'team': None,
                'positions': [],
                'frames_seen': 0
            }
        )
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("❌ Could not open video file")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps <= 0:
                fps = 30
            
            st.info(f"📊 Frames: {total_frames} | FPS: {fps:.1f}")
            
            # Progress UI
            progress_bar = st.progress(0)
            frame_counter = st.empty()
            placeholder = st.empty()
            status_text = st.empty()
            
            frame_count = 0
            
            # =========== PROCESS FRAMES ===========
            while cap.isOpened():
                
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize for speed
                frame = cv2.resize(frame, (1280, 720))
                
                # Process (frame stays BGR for detection)
                annotated_frame, players = tracker.process_frame(frame)
                
                # Convert to RGB for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Show preview
                placeholder.image(display_frame, use_container_width=True)
                
                # Update progress
                frame_count += 1
                progress = min(frame_count / max(1, total_frames), 1.0)
                progress_bar.progress(progress)
                frame_counter.text(
                    f"Frame {frame_count}/{total_frames} | Players: {len(players)}"
                )
                status_text.text(f"Processing... {progress*100:.1f}%")
            
            # Cleanup
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
        
        finally:
            try:
                os.unlink(video_path)
            except:
                pass
        
        # =========== RESULTS ===========
        st.success("✅ Complete!")
        
        st.subheader("📊 Player Statistics")
        
        if tracker.tracking_data:
            
            stats_df = tracker.get_stats()
            
            if not stats_df.empty:
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Download
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "player_stats.csv",
                    "text/csv"
                )
            else:
                st.warning("No statistics available")
        
        else:
            st.warning("⚠️ No players detected in video")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()
