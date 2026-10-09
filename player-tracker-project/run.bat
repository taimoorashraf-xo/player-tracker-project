@echo off
echo.
echo ======================================
echo    Player Tracker - Quick Start
echo ======================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting Streamlit app...
echo.
streamlit run player_tracker_final.py

pause
