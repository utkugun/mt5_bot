@echo off
cd /d "%~dp0"

rem Make sure the local LLM server is up before the bot starts polling
rem (config.LLM_PROVIDER = "ollama" by default -- harmless if already running).
rem OLLAMA_CONTEXT_LENGTH must be set before `ollama serve` starts -- Ollama's
rem OpenAI-compatible endpoint (which the bot uses) ignores any per-request
rem num_ctx option, so the server-wide default is the only way to raise it.
rem Without this it silently truncates input at 4096 tokens, which is smaller
rem than the trading system prompt + a full symbol snapshot, and the model
rem drops the forced tool call on a large fraction of cycles as a result.
set OLLAMA_CONTEXT_LENGTH=12000
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    start "" /min ollama serve
    timeout /t 3 /nobreak >nul
)

python -m mt5_bot.bot
pause
