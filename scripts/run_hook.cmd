: <<'CMDBLOCK'
@echo off
setlocal
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -X utf8 "%~dp0review_hook.py" %*
    exit /b %ERRORLEVEL%
)
py -3 -X utf8 "%~dp0review_hook.py" %*
exit /b %ERRORLEVEL%
CMDBLOCK
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$(dirname "$0")/review_hook.py" "$@"
fi
exec python "$(dirname "$0")/review_hook.py" "$@"
