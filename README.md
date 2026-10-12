# ⚽ Player Scouting & Tracking

> Find the right player for your team from match video and real match data. A university project prototype.

Upload match highlights (for example from YouTube), say what kind of player you need, and the app finds the players on screen, estimates how each one moves, ranks them against your requirement, and shows the best fits in a highlighted replay. A second tab scouts from real match data, judging players against **your team's formation and playstyle** and checking whether they would improve your current starters.

## What it does

- **🎥 Analyse a video:** detects and tracks players, estimates speed, distance, high-intensity running and pressing, groups them into teams by shirt colour, and ranks them with a photo for each player.
- **📊 Data scouting:** reads a team's formation and playstyle from real matches (free StatsBomb Open Data), finds the positions that need strengthening, and ranks candidates by role fit, style fit and whether they beat the current starter.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate                # Mac/Linux: source venv/bin/activate
pip install -r player-tracker-project/requirements.txt
.\start.bat                          # Mac/Linux: ./start.sh
```

The app opens at `http://localhost:8501`. Full instructions, settings and troubleshooting are in [`player-tracker-project/README.md`](player-tracker-project/README.md).

## Built with

Python · Streamlit · YOLOv8 · Deep SORT · OpenCV · EasyOCR · StatsBomb Open Data

## Status and limits

This is a working prototype for demonstration, not production software. Speed and distance from video are estimates, pressing is a proxy (the ball is not tracked), and passing and exact position cannot be measured from highlight clips. An optional demo mode adds clearly labelled simulated details; everything simulated is marked "(demo)" and never changes the fit score.
