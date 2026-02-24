@echo off
echo Running get_gold_prices.py in conda environment 'base'...
call conda run -n base python get_gold_prices.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running script. Please check if 'base' environment has akshare installed.
)
pause
