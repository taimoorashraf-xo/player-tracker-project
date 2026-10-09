# ⚽ Player Scouting & Tracking

Upload match highlights, say what kind of player you need, and the AI finds the players on screen, estimates how each one moves, ranks them against your requirement, and shows the best fits in the video.

This is a **prototype** that demonstrates what video analysis can do for scouting. It is not a replacement for professional tracking data.

## How it works

1. **Detect** - YOLOv8 finds the people in each frame; only people standing on the pitch are kept.
2. **Track** - Deep SORT follows each player. The tracker removes camera pans, so players keep their ID when the broadcast camera moves, and it starts afresh at every camera cut.
3. **Measure** - For each player who stays on screen long enough, the app estimates:
   - **Speed, top speed, distance and high-intensity running**, using the player's apparent height as the scale.
   - **Pressing activity**, the share of time spent closing in on an opponent (a proxy; the ball is not tracked).
4. **Group** - Players are split into teams by shirt colour.
5. **Rank** - Each measurement becomes a percentile among the players in the video, and the weighted average gives a fit score (0-100) for the requirement you choose: *High press*, *Counter-attack (pace)*, *Work rate (box-to-box)* or *Balanced athletic*. The weights can be changed.
6. **Show** - You get a ranked table with a photo of each player, a per-player report, a CSV download, and the video re-rendered with the best fits highlighted in gold.

### What it can and can't tell you

- Speed, distance and running intensity are **estimates from edited broadcast video**, not tracking-data accuracy. Vertical movement in the image is not corrected for camera angle, so running towards or away from the camera is under-estimated.
- **Pressing is a proxy.** The app does not know who has the ball.
- **Passing, dribbling success and exact position cannot be measured from highlights**, so players are ranked on movement and pressing only.
- A player seen in several camera shots appears once per shot, because players are not recognised across cuts yet.
- Shirt numbers are only readable in close-ups.
- Percentiles compare players **within the same video** only.

## Data scouting tab

A second tab ranks players using real match event data instead of video. Pick a position and a playing style (high press, possession build-up, counter-attack or balanced), and the app scores every player in the data against it and explains the fit.

- **Data:** free [StatsBomb Open Data](https://github.com/statsbomb/open-data) (selected men's competitions, including UEFA Euro 2020 and 2024). It downloads on first use (about a second per match), is cached in `data_cache/`, and needs an internet connection the first time.
- **Metrics (per 90 minutes):** passing (volume, completion, progressive, key, into the final third, long, crosses), pressing and defending, ball carrying, shooting, ball losses and average position.
- **Scoring:** percentiles within the same position group, weighted by the position and style. Every weight can be changed.
- **Limits:** event data has no distance run or sprint speed, so work rate and pace are proxies. A few matches means a small sample. Treat the result as a shortlist to watch.

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

From the folder that contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

*The first install takes several minutes.*

### 3. Run Application

Run from the same folder (the model file `yolov8m.pt` is loaded from the script's folder):

```bash
streamlit run player_tracker_final.py
```

Browser opens at: `http://localhost:8501`

## Usage

### Analyse a video

1. Open the **🎥 Analyse a video** tab and upload a highlights video (MP4, AVI, MOV, MKV).
2. Choose **What kind of player do you need?**
3. Click **Analyse video** and wait for the progress bar to finish.
4. Read the ranked table, choose which teams to scout, and open a **Player report**.
5. Set **Highlight the top N players** and watch the highlighted video. If you change the settings afterwards, click **Update highlighted video**.

#### Advanced settings

- **Detection model** - Nano is the fastest, Medium the most accurate. Nano and Small weights download automatically the first time you pick them.
- **Max video width** - Wider videos are scaled down. Smaller is faster.
- **Analyse every N frames** - 1 is the most precise and slowest. With 3, only every third frame is analysed.
- **Only players on the pitch** - Ignores people whose feet are not on grass. Turn it off for indoor or non-grass pitches.
- **Show live preview** - Off by default because sending frames to the browser slows processing.

Tips: shots that follow play for several seconds give the best measurements. Very short clips and fast replays give few measurable players.

### Data scouting

1. Open the **📊 Data scouting** tab, choose competitions and click **Load player data**.
2. Choose the position, the way your team plays, and a minimum number of minutes.
3. Read the ranked shortlist and pick a player for the fit report. Download the shortlist as CSV.

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```
Run this from the folder that contains `requirements.txt`, with your virtual environment active.

### Port 8501 Already in Use
```bash
streamlit run player_tracker_final.py --server.port 8502
```

### First Run is Slow
- Models download automatically the first time (YOLO weights, OCR weights)
- The first data-scouting load downloads match data, then it is cached

### "No player stayed on screen long enough to measure"
- Lower **Analyse every N frames** (for example 1 or 2)
- Try a larger detection model
- Turn off **Only players on the pitch** if the pitch is not grass
- Use a clip with longer continuous shots

### OCR Not Reading Shirt Numbers
- Numbers are only readable in close-ups; wide shots are too small
- OCR runs on a sample of frames per player and the most frequent reading is used
- If the OCR weights cannot be downloaded, shirt-number reading is disabled with a warning and everything else still works

## Project Structure

```
player-tracker-project/
├── player_tracker_final.py    (Main app: both tabs, detection and tracking)
├── video_metrics.py            (Speed, pressing, team grouping and ranking from video)
├── scouting.py                 (Data scouting metrics and fit scoring)
├── yolov8m.pt                  (YOLOv8 medium weights; nano/small download on first use)
├── data_cache/                 (Downloaded match data; created automatically)
├── requirements.txt            (Dependencies)
├── README.md                   (This file)
├── check_setup.py              (Verification script)
├── setup.py                    (Auto setup script)
├── run.sh / run.bat            (Start the app from a venv)
└── venv/                       (Virtual environment, created by you)
```

## Models and Methods

- **YOLOv8** - Person detection
- **Deep SORT** - Tracking, with shirt-colour histograms as appearance features
- **Camera-pan compensation** - Phase correlation between frames, so tracking and speed are measured in a stable coordinate system
- **Speed estimation** - Movement of the player's feet divided by the player's apparent height (assumed 1.8 m)
- **EasyOCR** - Shirt numbers in close-ups
- **K-means on shirt colour** - Team grouping

## Notes

- Tracking IDs are consistent for a player while they stay in one camera shot; after a cut the same player gets a new ID
- Processing time depends on the model size, video width, CPU/GPU and the "Analyse every N frames" setting
- Accuracy on real match footage has not been formally measured. The speed estimates were checked against synthetic scenes with known speeds (median error about 7%); real broadcast footage with zoom, blur and crowding will be less accurate

---

**Status:** Working prototype for demonstration.
