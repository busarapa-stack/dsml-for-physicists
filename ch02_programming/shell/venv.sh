#!/usr/bin/env bash
# Ch.2  code:venv -- create, activate, install, record, leave a virtual environment.
# Run line by line in a terminal (macOS/Linux). Windows PowerShell: see comments.
set -e
mkdir -p ~/physics-ch02

# 1. create
cd ~/physics-ch02
python -m venv .venv                 # on macOS with Homebrew use: python3 -m venv .venv

# 2. activate
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. install libraries
pip install numpy scipy matplotlib jupyter

# 4. record exact versions
pip freeze > requirements.txt

# 5. leave
deactivate
