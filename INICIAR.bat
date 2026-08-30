@echo off
setlocal
cd /d "C:\IA DE VOZ"
if not exist "C:\IA DE VOZ\.venv\Scripts\python.exe" (
  call "C:\IA DE VOZ\INSTALAR.bat"
)
set "HF_HOME=C:\IA DE VOZ\cache\huggingface"
set "HF_HUB_CACHE=C:\IA DE VOZ\cache\huggingface\hub"
set "TRANSFORMERS_CACHE=C:\IA DE VOZ\cache\transformers"
set "TORCH_HOME=C:\IA DE VOZ\cache\torch"
set "PIP_CACHE_DIR=C:\IA DE VOZ\cache\pip"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_TELEMETRY=1"
"C:\IA DE VOZ\.venv\Scripts\python.exe" "C:\IA DE VOZ\app.py"
