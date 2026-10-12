#!/usr/bin/env python3
"""
Automated setup script for Player Tracker
Run: python setup.py
"""

import subprocess
import sys
import os
import platform

HERE = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd, description=""):
    """Run shell command"""
    if description:
        print(f"\n▶ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("="*60)
    print("⚽ PLAYER TRACKER - AUTOMATED SETUP")
    print("="*60)
    print()
    
    system = platform.system()
    print(f"System: {system}")
    print(f"Python: {sys.version}")
    print()
    
    # Step 1: Upgrade pip
    print("STEP 1: Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install requirements
    print("\nSTEP 2: Installing packages...")
    print("⏳ This may take 5-10 minutes...")
    
    success = run_command(
        f"{sys.executable} -m pip install -r \"{os.path.join(HERE, 'requirements.txt')}\"",
        "Installing dependencies"
    )
    
    if not success:
        print("\n⚠️  Installation had issues. Trying alternative method...")
        run_command(
            f"{sys.executable} -m pip install -r \"{os.path.join(HERE, 'requirements.txt')}\" --no-cache-dir",
            "Retrying without cache"
        )
    
    # Step 3: Verify installation
    print("\nSTEP 3: Verifying installation...")
    run_command(f"{sys.executable} \"{os.path.join(HERE, 'check_setup.py')}\"", "Running diagnostics")
    
    # Success message
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next: Run the application")
    print()
    
    if system == "Windows":
        print("  streamlit run player_tracker_final.py")
    else:
        print("  python3 -m streamlit run player_tracker_final.py")
    
    print()
    print("Browser will open automatically at http://localhost:8501")
    print()

if __name__ == "__main__":
    main()
