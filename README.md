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

### Names, positions and demo mode

Highlights alone cannot tell the app who a player is, or what position they play. Two options fill that gap:

- **Your own input:** open "Edit names, positions and shirt numbers" and type them in. The **Position needed** filter then ranks only players tagged with that position (for example a right winger).
- **Demo mode:** one checkbox fills in simulated names, positions, shirt numbers and passing figures, so the whole flow can be shown on any clip. Everything simulated is labelled **(demo)**, appears in separate demo columns, is marked in the CSV, and **never changes the fit score**. If you present results from demo mode, say that those details are placeholders.

## Data scouting tab (team-aware)

A good scout does not ask "who is the best winger?" but "who is the best winger **for how we play**, and would he improve what we already have?". This tab works that way, using real match event data.

1. **Pick your team.** Choose a team from the data, and the app reads its **formation** (from the real starting line-ups) and its **playstyle** (pressing intensity, possession, directness, use of the wings, defensive line height and counter-attacks, each compared with the other teams). If your team is not in the data, **describe it by hand** with a formation and three sliders.
2. **See the squad needs.** For each position the formation uses, the app finds the current starter, judges him on *this* system, and shows where he ranks among all players in the data. The weakest position is selected for you automatically.
3. **Rank candidates for that position.** The fit score combines the **role** (what the position does in your formation; for example wing-backs in a back three are judged more on crossing and carrying) and the **style** (what your way of playing asks for, blended from your team's real style). Each candidate shows his role fit, style fit and **how much better or worse he is than your current starter** ("Clear upgrade", "Marginal upgrade" or "Not an upgrade"). Your own players are left out.

- **Data:** free [StatsBomb Open Data](https://github.com/statsbomb/open-data) (selected men's competitions, including UEFA Euro 2020 and 2024). It downloads on first use (about a second per match), is cached in `data_cache/`, and needs an internet connection the first time.
- **Player metrics (per 90 minutes):** passing, pressing and defending, ball carrying, shooting, ball losses and average position.
- **Scoring:** percentiles within the same position group, weighted by the position, the formation and the team's playstyle. Every weight can be changed.

### Edge cases the tab handles

| Situation | What happens |
|---|---|
| Your team is not in the data | Describe it by hand; you still get the style-aware ranking, without the squad analysis |
| The team used several formations | The most common one is used, and the others are listed |
| The team has played only a few matches | A warning says the playstyle estimate is rough |
| The formation has no slot for the position you chose (for example a winger in a 3-5-2) | A warning appears |
| The team has no qualifying starter in that position | The need is shown as high and no starter comparison is made |
| Players from your own team | Excluded from the candidates |
| Players who play several positions | Optional "can also play this position" setting; versatile players are labelled |
| Players with few minutes | Scores are pulled towards average (small-sample adjustment), and every player has a High / Medium / Low sample label |
| Older matches without pressing, carrying or xG data | Shown as n/a and counted as average, never as zero; mixing them in does not dilute other matches |
| Several competitions loaded | Players are compared across leagues of different strength, and this is disclosed |
| Goalkeepers | Not scored, because event metrics do not capture their job |

### Not handled yet

- Age, contract, cost, injuries and work permits (not in event data).
- Strength of the opposition a player faced, and match situations such as leading or trailing.
- Finer roles within a position (for example inverted versus traditional winger).
- Players who moved clubs mid-season are listed under their last team.
- Work rate and pace are proxies; event data has no distance run or sprint speed.

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

**Easiest:** from the top-level repository folder, run `.\start.bat` (Windows PowerShell) or `./start.sh` (Mac/Linux). It starts the app from the right folder.

**Or manually,** from the folder that contains `player_tracker_final.py`:

```bash
streamlit run player_tracker_final.py
```

Browser opens at: `http://localhost:8501`

## Usage

### Analyse a video

1. Open the **🎥 Analyse a video** tab and upload a highlights video (MP4, AVI, MOV, MKV).
2. Choose **What kind of player do you need?**
3. Click **Analyse video** and wait for the progress bar to finish.
4. Read the ranked table, choose which teams to scout, optionally tag names and positions (or switch on demo mode), and open a **Player report**.
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
2. Choose your team from the data (or describe it by hand) and read its formation, playstyle and squad needs.
3. Choose the position (your biggest need is pre-selected) and whether to scout for your team's playstyle or a different one.
4. Read the ranked shortlist, tick **Only show players who would improve on the current starter** if you like, and pick a player for the fit report. Download the shortlist as CSV.

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
├── team_context.py             (Team formation, playstyle, squad needs and upgrade logic)
├── demo_data.py                (Clearly labelled simulated placeholders for demos)
├── ui_theme.py                 (Design system: theme CSS and HTML components)
├── .streamlit/config.toml      (Dark theme; start the app from this folder so it is picked up)
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
