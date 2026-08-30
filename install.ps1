$ErrorActionPreference = 'Stop'
$Root = 'C:\IA DE VOZ'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
try { Start-Transcript -Path (Join-Path $Root 'logs\install.log') -Append | Out-Null } catch { }
if (-not (Test-Path -LiteralPath $Python)) {
    if (-not (Test-Path -LiteralPath 'C:\IA\python.exe')) { throw 'Python 3.11 não encontrado em C:\IA\python.exe' }
    & 'C:\IA\python.exe' -m venv (Join-Path $Root '.venv')
}
$env:HF_HOME = Join-Path $Root 'cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $Root 'cache\huggingface\hub'
$env:TRANSFORMERS_CACHE = Join-Path $Root 'cache\transformers'
$env:TORCH_HOME = Join-Path $Root 'cache\torch'
$env:PIP_CACHE_DIR = Join-Path $Root 'cache\pip'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
foreach ($d in @('input_audio','dataset','dataset\cleaned','dataset\segments','dataset\rejected','dataset\references','dataset\reports','training\runs','generated','logs','models','cache\huggingface','cache\transformers','cache\torch','cache\whisper','cache\pip','config','scripts')) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
& $Python -m pip install --upgrade pip
& $Python -m pip install 'torch==2.5.1+cu121' 'torchaudio==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
& $Python -m pip install -r (Join-Path $Root 'requirements.txt')
if (-not (Test-Path -LiteralPath (Join-Path $Root 'qwen3-tts\.git'))) { & git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git (Join-Path $Root 'qwen3-tts') }
& $Python -m pip install -e (Join-Path $Root 'qwen3-tts')
Write-Host "Instalação concluída em $Root"
try { Stop-Transcript | Out-Null } catch { }
