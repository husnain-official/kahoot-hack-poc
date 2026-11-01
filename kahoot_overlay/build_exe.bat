@echo off
setlocal
cd /d "%~dp0"

python -m pip install --upgrade pip pyinstaller
python -m pip install -r requirements.txt

pyinstaller --noconfirm --clean --onefile --noconsole --hidden-import pystray._win32 --name KahootOverlay main.pyw

echo.
echo Build complete: dist\KahootOverlay.exe
endlocal