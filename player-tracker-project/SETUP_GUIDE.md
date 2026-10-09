# ⚽ Player Tracking System - Complete Setup Guide

## Prerequisites
- Python 3.9+ installed
- 2GB free disk space (for models)
- 4GB+ RAM recommended

---

## Step-by-Step Setup

### Step 1: Extract the ZIP file

Extract all files to a folder:
```
player-tracker-project/
├── player_tracker_final.py
├── requirements.txt
├── README.md
├── check_setup.py
├── setup.py
└── SETUP_GUIDE.md
```

### Step 2: Open Terminal/Command Prompt

**Windows:** Press `Win+R`, type `cmd`, press Enter  
**Mac:** Open Applications → Utilities → Terminal  
**Linux:** Press Ctrl+Alt+T  

Navigate to extracted folder:
```bash
cd path/to/player-tracker-project
```

### Step 3: Create Virtual Environment

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

**Verify:** Command line should start with `(venv)`

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

⏳ **This takes 5-10 minutes** - Do NOT interrupt  

If error occurs:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Step 5: Verify Installation

```bash
python check_setup.py
```

Should show all ✅ marks

### Step 6: Run Application

```bash
streamlit run player_tracker_final.py
```

✅ Browser opens automatically at `http://localhost:8501`

If not, manually go to: http://localhost:8501

---

## Usage Guide

### Upload Video
1. Click **"Upload video"** box
2. Select MP4/AVI/MOV/MKV file
3. Click **"Process"** button

### Watch Live Preview
- Real-time bounding boxes around players
- Player IDs (consistent tracking)
- Detected shirt numbers
- Team colors

### View Results
- Statistics table shows:
  - Player ID
  - Shirt Number
  - Team
  - Frames Detected
  - Detection Rate (%)

### Download Data
- Click **"📥 Download CSV"** to save results

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'X'"

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### ❌ "Python version is X.Y - Need 3.9+"

**Solution:** Install Python 3.9+
- Go to python.org
- Download Python 3.9 or higher
- Reinstall with "Add to PATH" checked

### ❌ "Port 8501 already in use"

**Solution:**
```bash
streamlit run player_tracker_final.py --server.port 8502
```

Then go to: http://localhost:8502

### ❌ "First run is very slow"

**This is normal!**
- Models downloading (~700MB)
- First run: 5-10 minutes
- Subsequent runs: Faster (cached)
- Do NOT close or interrupt

### ❌ "CUDA out of memory"

**This is OK** - App uses CPU  
Just wait, it will work fine

### ❌ "Cannot find libGL.so" (Linux)

**Solution:**
```bash
sudo apt-get install libgl1-mesa-glx
```

### ❌ "Cannot find libomp" (Mac)

**Solution:**
```bash
brew install libomp
```

### ❌ "No players detected"

**Possible reasons:**
- Video quality too low
- People too small in frame
- Low lighting
- Try different video

**Try:** Lower confidence threshold  
Edit line in `player_tracker_final.py`:
```python
conf=0.4  # Change to 0.3
```

### ❌ "Shirt numbers not being read"

**Limitations:**
- Works best with clear, straight-on numbers
- Depends on jersey quality
- Lighting affects accuracy
- This is normal OCR behavior

---

## Folder Structure After Setup

```
player-tracker-project/
├── player_tracker_final.py     (Main app)
├── requirements.txt             (Dependencies)
├── README.md                    (Overview)
├── SETUP_GUIDE.md              (This file)
├── check_setup.py              (Verify script)
├── setup.py                    (Auto setup)
├── venv/                       (Virtual environment - created)
└── player_stats.csv            (Output - created after processing)
```

---

## Quick Reference

### Activate Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Run App
```bash
streamlit run player_tracker_final.py
```

### Deactivate Environment
```bash
deactivate
```

### Check Dependencies
```bash
python check_setup.py
```

### Auto Setup (Alternative)
```bash
python setup.py
```

---

## Performance Tips

1. **Smaller videos process faster**
   - 1 min video ≈ 2-3 min processing
   - 5 min video ≈ 10-15 min processing

2. **Optimal resolution: 720p**
   - Too small: Poor detection
   - Too large: Slow processing

3. **Clear visibility matters**
   - Good lighting helps OCR
   - Shirt numbers must be visible
   - Players shouldn't be too far away

4. **System requirements**
   - RAM: 4GB minimum
   - Disk: 2GB for models
   - CPU: Modern multi-core recommended

---

## Video Specifications

**Supported Formats:**
- MP4, AVI, MOV, MKV, FLV

**Recommended:**
- Resolution: 720p (1280×720)
- Codec: H.264
- Frame Rate: 24-60 FPS
- Duration: 1-5 minutes
- File Size: <500MB

**Works With:**
- Smartphone recordings
- CCTV footage
- Sports videos
- Screen recordings

---

## Success Indicators

✅ You'll know it's working when:
- Streamlit UI appears in browser
- Video upload box is visible
- "Process" button is clickable
- Live preview shows bounding boxes
- Progress bar updates
- Player IDs appear consistently
- Stats table shows at the end
- CSV download works

---

## Getting Help

1. **Check Python version:** `python --version`
2. **Verify environment:** Look for `(venv)` prefix
3. **Run diagnostics:** `python check_setup.py`
4. **Reinstall cleanly:**
   ```bash
   deactivate
   rmdir venv
   python -m venv venv
   # Activate again
   pip install -r requirements.txt --no-cache-dir
   ```

---

## System Compatibility

| OS | Status | Notes |
|-----|--------|-------|
| Windows 10/11 | ✅ Tested | Works great |
| macOS 11+ | ✅ Tested | Works great |
| Ubuntu 20.04+ | ✅ Tested | Works great |
| Raspberry Pi | ⚠️ Slow | Possible but very slow |

---

**You're ready to track players!** 🚀

Questions? Check README.md or run `python check_setup.py`
