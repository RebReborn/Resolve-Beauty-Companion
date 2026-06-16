@echo off
:: Navigate to the script's directory
cd /d "%~dp0"

:: Create Desktop Shortcut if it doesn't exist
set SHORTCUT_PATH="%USERPROFILE%\Desktop\Resolve Beauty Companion.lnk"
if not exist %SHORTCUT_PATH% (
    echo Creating Desktop Shortcut...
    powershell -NoProfile -Command ^
        "$WshShell = New-Object -ComObject WScript.Shell; " ^
        "$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Resolve Beauty Companion.lnk'); " ^
        "$Shortcut.TargetPath = '%~dp0run_app.bat'; " ^
        "$Shortcut.WorkingDirectory = '%~dp0'; " ^
        "$Shortcut.IconLocation = '%~dp0icon.png'; " ^
        "$Shortcut.Save()"
)

:: Run the PyQt6 application silently in the background (no cmd window)
start "" "venv\Scripts\pythonw.exe" main.py

exit
