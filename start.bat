@echo off
rem Starts the app from the right folder, whichever folder you run this from.
cd /d "%~dp0player-tracker-project"
python -m streamlit run player_tracker_final.py
pause
