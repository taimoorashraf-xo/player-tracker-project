#!/bin/bash
# Starts the app from the right folder, whichever folder you run this from.
cd "$(dirname "$0")/player-tracker-project" || exit 1
python3 -m streamlit run player_tracker_final.py
