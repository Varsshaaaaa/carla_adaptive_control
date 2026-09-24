@echo off
echo Starting CARLA Server with low quality settings...
:: Adjust the path below if your CarlaUE4.exe is in a different location.
:: Assuming the standard CARLA 0.9.15 Windows build folder structure.
set CARLA_EXE="D:\WindowsNoEditor\CarlaUE4.exe"

if exist %CARLA_EXE% (
    %CARLA_EXE% -windowed -ResX=800 -ResY=600 -quality-level=Low
) else (
    echo CarlaUE4.exe not found at %CARLA_EXE%.
    echo Please make sure you extracted CARLA to D:\WindowsNoEditor
    pause
)
pause
