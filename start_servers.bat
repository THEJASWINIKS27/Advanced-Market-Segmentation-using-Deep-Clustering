@echo off
REM Start backend server
start cmd /k "python backend.py"
timeout /t 5

REM Start frontend server
start cmd /k "python -m streamlit run frontend.py"
