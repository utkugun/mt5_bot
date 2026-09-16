@echo off
REM Starts the whole PO PDF -> Excel stack from a clean slate:
REM   1. Open WebUI          (http://localhost:8080)
REM   2. PO PDF Extractor    OpenAPI tool server (port 8011) - the reliable one, used by the "den" model
REM   3. PO PDF Extractor    MCP tool server (port 8010) - kept for reference; currently disabled in Open WebUI
REM   4. Excel MCP Server    (haris-musa/excel-mcp-server) streamable-http (port 8008)
REM Each opens in its own window so you can watch its logs / close it independently.

set VENV_PY="C:\Users\ugun\Documents\utku python\open-webui\venv\Scripts\python.exe"
set VENV_SCRIPTS=C:\Users\ugun\Documents\utku python\open-webui\venv\Scripts
set OPENWEBUI_EXE="C:\Users\ugun\Documents\utku python\open-webui\venv\Scripts\open-webui.exe"
set PO_MCP_DIR="C:\Users\ugun\Documents\utku python\po_mcp"

echo Starting Open WebUI (http://localhost:8080) ...
start "Open WebUI" cmd /k %OPENWEBUI_EXE% serve

echo Starting PO PDF Extractor - OpenAPI (port 8011) ...
start "PO PDF Extractor (OpenAPI 8011)" cmd /k "cd /d %PO_MCP_DIR% && %VENV_PY% server.py --mode openapi --port 8011"

echo Starting PO PDF Extractor - MCP (port 8010, disabled in Open WebUI by default) ...
start "PO PDF Extractor (MCP 8010)" cmd /k "cd /d %PO_MCP_DIR% && %VENV_PY% server.py --mode mcp --port 8010"

echo Starting Excel MCP Server (port 8008) ...
start "Excel MCP Server (8008)" cmd /k "set EXCEL_FILES_PATH=C:\Users\ugun\Documents\utku python&& set FASTMCP_HOST=127.0.0.1&& set FASTMCP_PORT=8008&& "%VENV_SCRIPTS%\excel-mcp-server.exe" streamable-http"

echo.
echo All four services are starting in separate windows.
echo Open WebUI needs ~10-20s to be ready at http://localhost:8080
echo Close this window any time - the servers keep running in their own windows.
pause
