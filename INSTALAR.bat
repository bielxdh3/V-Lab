@echo off
setlocal
cd /d "C:\IA DE VOZ"
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\IA DE VOZ\install.ps1"
if errorlevel 1 (echo Falha na instalacao. Veja C:\IA DE VOZ\logs e pause & exit /b 1)
echo OK. Execute INICIAR.bat
pause
