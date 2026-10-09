# ⚽ Player Tracking System

Real-time player detection, tracking, and analysis from video files.

## Features

✅ **Real-time Detection** - YOLOv8 person detection  
✅ **Persistent Tracking** - Deep SORT multi-object tracking  
✅ **Shirt Number OCR** - Automatic shirt number extraction  
✅ **Team Detection** - Color-based team classification  
✅ **Live Preview** - Watch detection as it happens  
✅ **Statistics** - Frame counts, detection rates, CSV export  
✅ **Speed control** - Process every Nth frame to trade precision for speed  

## Requirements

- Python 3.9+
- 2GB free disk space (for models)
- 4GB+ RAM recommended
- A GPU is optional; without one the app runs on CPU and is slow on long videos

## Quick Start

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*First install takes 5-10 minutes (downloads models)*

### 3. Run Application

Run from this folder (the model file `yolov8m.pt` is loaded from the script's folder):

```bash
streamlit run player_tracker_final.py
```

Browser opens at: `http://localhost:8501`

## Usage

1. **Upload Video** - MP4/AVI/MOV/MKV format
2. **Click Process** - App detects and tracks players
3. **Watch Live Preview** - See bounding boxes in real-time
4. **View Statistics** - Get final player stats
5. **Download CSV** - Export results

Use **Process every N frames** to speed things up. The default of 3 processes every third frame. Set it to 1 for maximum precision, which is much slower on CPU.

## Video Requirements

- **Format:** MP4, AVI, MOV, MKV
- **Resolution:** 480p-1080p recommended
- **Duration:** 1-5 minutes ideal
- **Codec:** H.264 or similar

## Output Statistics

- Player ID (persistent tracking)
- Shirt Number (OCR detected)
- Team (color detected)
- Frames Detected (appearance count)
- Detection Rate (%)

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### `pkg_resources` / `No module named 'pkg_resources'`
The tracker's dependency `deep-sort-realtime` needs `setuptools<81`. It is pinned in `requirements.txt`; re-run `pip install -r requirements.txt` if you installed before this was added.

### Port 8501 Already in Use
```bash
streamlit run player_tracker_final.py --server.port 8502
```

### First Run is Slow
- Normal behavior, models download automatically
- Subsequent runs are faster (cached)

### No Players Detected
- Try a different video with better visibility
- Ensure people are clearly visible in frame
- Check video quality
- Note that YOLO only detects players at a confidence of 0.4 or higher; very small or blurred players may be missed

### OCR Not Reading Shirt Numbers
- Works best with clear, straight-on numbers
- Lighting affects accuracy
- OCR runs on a sample of frames per player (up to 10 attempts), and the most frequent reading is reported
- The first run downloads the EasyOCR weights; if they cannot be downloaded, shirt-number OCR is disabled with a warning and tracking still works

## Project Structure

```
player-tracker-project/
├── player_tracker_final.py    (Main app)
├── yolov8m.pt                  (YOLOv8 medium weights)
├── requirements.txt            (Dependencies)
├── README.md                   (This file)
├── check_setup.py              (Verification script)
├── setup.py                    (Auto setup script)
├── run.sh / run.bat            (Start the app from a venv)
└── venv/                       (Virtual environment, created by you)
```

## Models Used

- **YOLOv8 Medium** - Person detection
- **EasyOCR** - Shirt number extraction
- **DeepSort** - Player tracking
- **MobileNet** (via DeepSort) - Appearance features for re-identification

## Performance

Processing time depends heavily on resolution, CPU/GPU and the "Process every N frames" setting. Measure on your own machine before relying on a time estimate. Each processed frame runs YOLOv8m detection plus DeepSort, so CPU-only runs on long videos take a while.

## Notes

- Tracking IDs are consistent for the same player across frames while they stay in view; a player who leaves the frame for long may get a new ID
- Team detection reads the colour of the shirt area (upper body), so it depends on jersey colours and lighting
- OCR accuracy improves with clear shirt numbers
- Works on CPU (GPU optional)

## Support

Check SETUP_GUIDE.md for detailed troubleshooting.

---

**Status:** Working prototype. Tested end to end on synthetic video; accuracy on real match footage has not been measured.
