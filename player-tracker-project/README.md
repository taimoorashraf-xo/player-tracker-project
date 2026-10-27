# Player Scout

Two ways to judge a footballer, in one Streamlit app.

- **Clip analysis** - upload match video and measure how every player on the pitch moves:
  top speed, distance, high-intensity running, sprints, accelerations and pressing.
- **Data scouting** - rank players on real match events against your team's formation and
  playing style, and see whether they would improve your current starter.

A prototype built for a university project. The movement numbers are estimates from
broadcast video, not professional tracking data, and the app says so where it matters.

## How the clip analysis works

1. **Detect.** YOLOv8 finds people in each analysed frame. Anyone whose feet are not on
   grass is dropped, along with boxes too small, too large or too wide to be one player.
2. **Remove the camera.** Features on the pitch (never on the players) are tracked between
   frames and fitted with a similarity transform, so pan, tilt, rotation **and zoom** are
   taken out before anything is measured. Cuts to a new shot are detected from the layout
   of the picture, and brief losses of tracking are bridged rather than treated as cuts.
3. **Follow the players.** Deep SORT tracks each player in the shot's stabilised frame,
   using kit colour as the appearance cue.
4. **Convert pixels to metres.** The apparent height of every player seen is used to fit
   how many pixels make a metre at each row of the image. Movement across the picture and
   movement into the distance are then converted separately, because they do not share a
   scale.
5. **Measure and rank.** Paths are smoothed, turned into speed and acceleration, and scored
   against the kind of player you asked for.

### Why the geometry matters

A player running away from the camera covers far more ground than the same number of pixels
would suggest nearer the touchline. Measuring both directions with a single scale, which is
the obvious approach, badly under-reads anything other than running across the camera.

Against a simulated pinhole camera with known player speeds:

| Running direction | Single-scale approach | This app |
|---|---|---|
| Across the camera | 0% error | +1% |
| 45 degrees | −36% | +2% |
| Straight away from the camera | −76% | +2% |

End to end on a simulated clip with a panning, zooming camera and a cut, measuring twelve
players of known speed, the median error falls from **43% to 11%**. Under a 35% zoom a
stationary player drifted 41 px with the previous translation-only camera model and 0.7 px
with this one.

### What it cannot do

- **Distance rests on assumptions.** Players are assumed to be 1.80 m tall on a flat pitch.
- **Depth needs a field of view.** Player heights fix the scale across the picture but say
  nothing about how far away things are; that needs the camera's field of view, which is
  not visible in the footage. The app assumes a typical broadcast framing and lets the
  movement in the clip nudge it within bounds. Runs towards or away from the camera carry
  more uncertainty than runs across it.
- **Pressing is a proxy.** It counts time spent closing down the nearest opponent. The ball
  is not tracked, so it cannot know who had it.
- **A player in two shots is counted twice** unless their shirt number is readable in both.
  Teammates wear the same kit, so nothing else in the picture proves they are the same
  person, and guessing would silently merge two players' numbers.
- **Passing, dribbling and pitch position cannot be measured** from highlights at all. Use
  the data scouting tab for those.
- **Sprint and acceleration counts** are left blank for players measured for under 8
  seconds, because one burst in a short spell scales up to a meaningless per-minute figure.

The app warns on screen when the footage itself made things shakier: camera movement that
could not be followed, shots too short to measure, too few players to judge perspective, or
an implied camera height unlike a real broadcast camera.

## Data scouting

Pick a team and the app reads its formation from real starting line-ups and its playing
style from its matches - pressing intensity, possession, directness, use of the wings and
defensive line - each compared against the other teams in the data. It then finds which
positions the current starters are weakest in for that system, and ranks candidates on role
fit, style fit and the gap to the incumbent. If your team is not in the data, describe it by
hand instead.

Data: free [StatsBomb Open Data](https://github.com/statsbomb/open-data), downloaded once
per match and cached in `data_cache/`. Work rate and pace there are proxies too: event data
holds no distance run or sprint speed. Goalkeepers are not scored.

## Running it

```bash
python -m venv venv
venv\Scripts\activate                # Mac/Linux: source venv/bin/activate
pip install -r player-tracker-project/requirements.txt
```

Then from the repository root:

```bash
.\start.bat                          # Mac/Linux: ./start.sh
```

The app opens at `http://localhost:8501`. Detector weights download on first use.

**Clips of 20-60 seconds that follow play work best.** Fast cuts and close-ups give the
tracker little to work with. Lower "analyse 1 frame in" for more accuracy and less speed;
above 3 the speeds become unreliable.

## Layout

```
player-tracker-project/
├── player_tracker_final.py   Streamlit interface
├── analysis.py               detection, tracking and the whole-clip pipeline
├── tracking.py               camera motion, perspective scale, re-identification
├── video_metrics.py          speed, distance, sprints, pressing, ranking
├── ui_theme.py               stylesheet, movement map and speed trace
├── scouting.py               match-event metrics and fit scoring
├── team_context.py           formation, playing style and squad needs
├── demo_data.py              clearly labelled placeholder names
├── check_setup.py            dependency check
└── requirements.txt
```

## Demo mode

Highlights cannot tell you who a player is. Demo mode fills in invented names and shirt
numbers so the whole flow can be shown; every one is marked "(demo)" and none of them
touches a score. Say so if you present results from it.
