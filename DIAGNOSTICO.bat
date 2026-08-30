@echo off
setlocal
cd /d "C:\IA DE VOZ"
if not exist "C:\IA DE VOZ\.venv\Scripts\python.exe" call "C:\IA DE VOZ\INSTALAR.bat"
"C:\IA DE VOZ\.venv\Scripts\python.exe" -c "from app import diagnose_system; print(diagnose_system())"
pause
