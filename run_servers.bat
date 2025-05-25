@echo off
start cmd /k "python backend.py"
timeout /t 5
start cmd /k "streamlit run frontend.py"
