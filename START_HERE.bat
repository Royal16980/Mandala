@echo off
color 0A
echo ========================================
echo   MANDALA LASER ENGRAVING APP
echo   Enhanced Version - 20 Layers
echo ========================================
echo.
echo Starting application...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Try to use the correct Python
python -m streamlit run app_enhanced.py

pause
