# ⚽ Player Tracking System

Real-time player detection, tracking, and analysis from video files.

## Features

✅ **Real-time Detection** - YOLOv8 person detection  
✅ **Persistent Tracking** - Deep SORT multi-object tracking  
✅ **Shirt Number OCR** - Automatic shirt number extraction  
✅ **Team Detection** - Color-based team classification  
✅ **Live Preview** - Watch detection as it happens  
✅ **Statistics** - Frame counts, detection rates, CSV export  

## Requirements

- Python 3.9+
- 2GB free disk space (for models)
- 4GB+ RAM recommended

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

### Port 8501 Already in Use
```bash
streamlit run player_tracker_final.py --server.port 8502
```

### First Run is Slow
- Normal behavior, models download automatically
- Subsequent runs are faster (cached)

### No Players Detected
- Try different video with better visibility
- Ensure people are clearly visible in frame
- Check video quality

### OCR Not Reading Shirt Numbers
- Works best with clear, straight-on numbers
- Lighting affects accuracy
- This is a known OCR limitation

## Project Structure

```
player-tracker-project/
├── player_tracker_final.py    (Main app)
├── requirements.txt            (Dependencies)
├── README.md                   (This file)
├── check_setup.py             (Verification script)
├── setup.py                   (Auto setup script)
└── venv/                      (Virtual environment)
```

## Models Used

- **YOLOv8 Medium** - Person detection
- **EasyOCR** - Shirt number extraction
- **DeepSort** - Player tracking
- **MobileNet** - Feature extraction

## Performance

- 1-minute video: ~2-3 minutes processing
- 5-minute video: ~10-15 minutes processing
- Depends on resolution and CPU

## Notes

- Tracking ID is consistent for same player across frames
- Team detection accuracy depends on jersey colors
- OCR accuracy improves with clear shirt numbers
- Works on CPU (GPU optional)

## Support

Check SETUP_GUIDE.md for detailed troubleshooting.

---

**Status:** Production Ready ✅
