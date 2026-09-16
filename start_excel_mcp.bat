@echo off
REM Starts the Excel MCP tool server (haris-musa/excel-mcp-server), Streamable HTTP,
REM on port 8008 - same URL Open WebUI already has registered as a Tool Server:
REM   Admin Panel > Settings > Tools > Add Tool Server > Type: MCP > URL: http://127.0.0.1:8008/mcp
REM
REM EXCEL_FILES_PATH is the root directory the server is allowed to read/write.
REM All tool calls use paths relative to this root (e.g. "po ozet\\file.xlsx").
set VENV_SCRIPTS=C:\Users\ugun\Documents\utku python\open-webui\venv\Scripts
set EXCEL_FILES_PATH=C:\Users\ugun\Documents\utku python
set FASTMCP_HOST=127.0.0.1
set FASTMCP_PORT=8008

echo Starting Excel MCP Server (streamable-http, port 8008) ...
"%VENV_SCRIPTS%\excel-mcp-server.exe" streamable-http
