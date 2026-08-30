<div align="center">

# V-Lab

**Voice Lab**

Local-first voice cloning, dataset preparation, fine-tuning, and multi-voice experimentation with Qwen3-TTS.

[![Status](https://img.shields.io/badge/status-active%20development-orange)](#project-status)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D4)](#requirements)
[![Runtime](https://img.shields.io/badge/runtime-Python%203.11-3776AB)](#technology)
[![TTS](https://img.shields.io/badge/TTS-Qwen3--TTS-6C5CE7)](#models)
[![Interface](https://img.shields.io/badge/interface-Gradio-FF7C00)](#technology)
[![Privacy](https://img.shields.io/badge/privacy-local--first-2EA44F)](#local-first-boundary)

V-Lab turns raw recordings into isolated voice profiles that can be cleaned, transcribed, prepared, cloned, evaluated, and trained without sending the user's voice to a hosted speech service.

</div>

> [!IMPORTANT]
> V-Lab is experimental software under active development. Voice cloning and zero-shot generation are functional; fine-tuning support still depends heavily on the selected Qwen3-TTS model, available VRAM, and upstream trainer behavior.

## The idea at a glance

```text
                         ┌──────────────────────┐
                         │     Raw recordings   │
                         │ wav · mp3 · flac ... │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Voice profile     │
                         │ speaker · reference  │
                         │ private local data   │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │     Dataset preparation     │
                     │ inspect · segment · clean   │
                     │ transcribe · score · review │
                     └──────────────┬──────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │        Qwen3-TTS pipeline       │
                    │ tokenizer · audio codes · clone │
                    │ training preflight · checkpoints│
                    └───────────┬───────────┬─────────┘
                                │           │
                      zero-shot │           │ fine-tuning
                                │           │
                     ┌──────────▼───┐   ┌──▼──────────────┐
                     │ Voice output │   │ Training lineage │
                     │ generated wav│   │ runs · checkpoints│
                     └──────────────┘   └──────────────────┘
```

The base TTS models are shared. Speaker-specific recordings, datasets, references, training runs, checkpoints, and generated audio remain isolated inside their own voice profile.

## What V-Lab does

V-Lab is built around a simple workflow: give each person their own profile, drop in recordings, prepare the dataset, review what the pipeline produced, and use that voice independently.

### Voice profiles

Each person has an isolated workspace containing their own:

- input recordings;
- cleaned and segmented clips;
- transcripts and metadata;
- accepted and rejected samples;
- reference audio;
- Qwen-compatible dataset files;
- training runs and checkpoints;
- evaluation artifacts;
- generated speech.

Switching the active voice switches the entire speaker-sensitive context of the application.

### Automatic dataset preparation

The preparation pipeline can handle the repetitive work required before cloning or training:

- inspect source audio and supported formats;
- convert files into a consistent working representation;
- segment long recordings with voice activity detection;
- apply conservative audio cleanup;
- transcribe speech locally;
- score and classify clips for review;
- choose a speaker reference;
- build Qwen3-TTS JSONL data;
- generate tokenizer audio codes;
- keep rejected or questionable clips available for inspection.

The goal is not to erase every imperfection from a recording. V-Lab prioritizes preserving the characteristics that make a voice sound like itself.

### Review before training

Prepared clips remain inspectable instead of disappearing into an opaque training job.

The review workflow exposes:

- original and processed audio;
- transcription;
- clip duration;
- quality classification;
- rejection or review reason;
- confidence and audio-quality metadata where available.

Accepted data can then be rebuilt into the active profile's training dataset.

## Models

V-Lab currently targets the Qwen3-TTS Base family.

| Model | Primary use | Current role |
|---|---|---|
| **Qwen3-TTS 1.7B Base** | Highest-quality local cloning path | Main model |
| **Qwen3-TTS 0.6B Base** | Lighter inference and experimentation | Low-resource alternative |
| **Qwen3-TTS Tokenizer 12Hz** | Dataset audio-code generation | Dataset preparation |

The application keeps model assets shared instead of copying several gigabytes of base weights into every voice profile.

## How a voice workflow works

```text
Create / select voice
        │
        ▼
Add recordings
        │
        ▼
Prepare dataset
        │
        ├── inspect
        ├── segment
        ├── clean
        ├── transcribe
        ├── classify
        ├── select reference
        └── tokenize
        │
        ▼
Review questionable clips
        │
        ▼
Rebuild accepted dataset
        │
        ├──────────────► Zero-shot clone
        │
        ▼
Training preflight
        │
        ▼
Fine-tuning / checkpoint workflow
        │
        ▼
Generate and compare speech
```

## Project status

The current development baseline includes:

- [x] local Gradio dashboard;
- [x] Windows hardware and CUDA diagnostics;
- [x] multiple isolated voice profiles;
- [x] per-profile input folders and dataset storage;
- [x] local audio conversion and segmentation;
- [x] local transcription with Faster-Whisper;
- [x] clip quality classification and review workflow;
- [x] automatic reference-audio selection;
- [x] Qwen3-TTS JSONL generation;
- [x] Qwen tokenizer `audio_codes` preparation;
- [x] Qwen3-TTS 1.7B zero-shot inference;
- [x] Qwen3-TTS 0.6B inference path;
- [x] training configuration and hardware preflight;
- [x] profile-scoped training runs, checkpoints, and generated output;
- [x] local-only voice-data boundary;
- [ ] fully validated low-VRAM 1.7B fine-tuning;
- [ ] fully validated LoRA/PEFT workflow;
- [ ] robust automatic multi-speaker diarization;
- [ ] broader GPU validation.

> [!NOTE]
> V-Lab intentionally treats model loading, inference support, and fine-tuning support as different capability levels. A model being able to generate speech on a GPU does not imply that full training will fit on the same GPU.

## Technology

| Layer | Responsibility | Technology |
|---|---|---|
| Interface | Local dashboard and workflow controls | Gradio |
| Runtime | Application logic and orchestration | Python 3.11 |
| Deep learning | TTS inference and training runtime | PyTorch + CUDA |
| Voice model | Cloning and speech generation | Qwen3-TTS |
| ASR | Local Portuguese transcription | Faster-Whisper |
| Audio tooling | Conversion and media handling | FFmpeg |
| Segmentation | Speech-region detection | VAD pipeline |
| Model ecosystem | Model/config loading | Transformers + Hugging Face |
| Storage | Voice profiles, datasets, runs, outputs | Local filesystem |

## Requirements

Current target platform:

- Windows 10 or Windows 11, 64-bit;
- Python 3.11 environment managed by the project;
- enough free disk space for Qwen models, ASR models, datasets, and checkpoints;
- NVIDIA GPU recommended for practical Qwen3-TTS inference;
- CUDA-capable PyTorch build compatible with the installed GPU;
- sufficient system RAM for model loading and CPU/GPU offload.

Fine-tuning requires substantially more memory than inference.

> [!WARNING]
> GPU support is hardware-dependent. V-Lab performs a preflight before training rather than assuming that a model which fits for inference will also fit for backpropagation.

## Quick start

### 1. Clone the repository

```powershell
git clone https://github.com/bielxdh3/V-Lab.git
cd V-Lab
```

### 2. Install the local environment

Use the project installer:

```powershell
.\INSTALAR.bat
```

The installer prepares the Python environment and required local dependencies.

### 3. Start V-Lab

```powershell
.\INICIAR.bat
```

The launcher starts the local application and opens the dashboard in the browser.

### 4. Create a voice

In the **Voices** section:

1. create a new voice profile;
2. select it as the active voice;
3. open its audio folder;
4. copy the recordings into that profile.

### 5. Prepare the dataset

Open **Prepare Dataset** and run the automatic pipeline.

Review clips marked for manual attention before starting a serious training run.

### 6. Generate or train

Use:

- **Test Voice** for zero-shot generation;
- **Training** for supported fine-tuning workflows.

## Repository and data layout

The code repository should stay separate from large or private runtime data.

```text
V-Lab/
├── app.py / app/                 Application and GUI code
├── config/                       Runtime configuration
├── tools/                        Local helpers and audio tooling
├── scripts/                      Installation / validation helpers
├── INSTALAR.bat                  Environment setup
├── INICIAR.bat                   Application launcher
├── DIAGNOSTICO.bat               Hardware/runtime diagnostics
└── README.md

Runtime data/
├── models/                       Shared base models
├── cache/                        Hugging Face / Torch / ASR caches
└── voices/
    └── <voice_id>/
        ├── profile.json
        ├── input_audio/
        ├── dataset/
        ├── training/
        ├── evaluation/
        └── generated/
```

Exact runtime paths may differ by installation, but speaker-specific artifacts must remain isolated by `voice_id`.

## Voice isolation

V-Lab does **not** treat several people as one giant speaker dataset.

Each voice profile has its own training lineage.

```text
Shared base model
      │
      ├── Voice A
      │   ├── reference
      │   ├── dataset
      │   ├── runs
      │   └── generated audio
      │
      ├── Voice B
      │   ├── reference
      │   ├── dataset
      │   ├── runs
      │   └── generated audio
      │
      └── Voice C
          └── ...
```

Cross-profile references, dataset rows, checkpoints, and generated outputs are treated as isolation violations instead of being silently accepted.

## Local-first boundary

Voice data is especially sensitive, so V-Lab treats local processing as an architectural boundary.

- source recordings stay on the local machine;
- transcription runs locally;
- Qwen3-TTS inference runs locally;
- dataset preparation runs locally;
- training runs locally when the hardware supports it;
- the application does not require OpenAI, ElevenLabs, Google Speech, Azure Speech, or another hosted voice API;
- voice datasets, generated audio, model caches, and checkpoints should remain outside Git;
- the public repository should contain code and reproducible configuration, not personal voice data.

> [!CAUTION]
> Do not commit real recordings, private transcripts, datasets, generated personal speech, checkpoints containing speaker adaptation, or model caches to the repository.

## Hardware behavior

V-Lab can use system RAM as part of model loading and CPU/GPU offload when VRAM is constrained.

That means a generation may spend significant time preparing or moving model components before the GPU-heavy portion begins.

The current development environment has demonstrated that:

- 1.7B inference can work under tight VRAM with offload;
- 0.6B can provide a lighter inference path;
- training memory requirements are much higher than inference requirements;
- a low-VRAM GPU may be able to generate successfully while still being rejected by the training preflight.

This behavior is expected and should not be interpreted as a guarantee for every GPU.

## Validation

Useful validation areas include:

- CUDA availability and a real GPU tensor operation;
- audio reading and FFmpeg conversion;
- segmentation;
- local ASR;
- dataset JSONL validation;
- Qwen tokenizer audio-code generation;
- zero-shot inference;
- profile-isolation checks;
- training preflight;
- checkpoint loading when supported.

Use the project diagnostic launcher for the current system:

```powershell
.\DIAGNOSTICO.bat
```

## Current limitations

- full Qwen3-TTS 1.7B fine-tuning is still highly VRAM-sensitive;
- Qwen3-TTS 0.6B training may depend on upstream trainer compatibility rather than model size alone;
- LoRA/PEFT is not yet treated as a universally validated production path;
- automatic diarization of several speakers inside one recording is not yet a guaranteed boundary;
- model switching on constrained GPUs can involve long cold-load/offload phases;
- training quality still depends heavily on dataset quality, transcription accuracy, and speaker consistency;
- Windows is the current supported platform;
- this is not a hosted service or a finished consumer release.

## Roadmap

- [ ] validate a reliable low-VRAM fine-tuning path for Qwen3-TTS 1.7B;
- [ ] validate LoRA/PEFT adapters per voice profile;
- [ ] improve automatic speaker-contamination detection;
- [ ] add stronger dataset quality metrics and checkpoint comparison;
- [ ] expand hardware profiling and VRAM estimation;
- [ ] improve model residency and switching behavior;
- [ ] add reproducible packaging without bundling private voice data;
- [ ] broaden validation across modern NVIDIA GPUs;
- [ ] continue refining the dashboard without turning it into a visual maze.

## Upstream projects

V-Lab builds on open-source projects including:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for speech synthesis and voice cloning;
- [Gradio](https://github.com/gradio-app/gradio) for the local web interface;
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for local transcription;
- [FFmpeg](https://ffmpeg.org/) for audio conversion and media handling;
- [PyTorch](https://pytorch.org/) for GPU-accelerated inference and training.

V-Lab is an independent project and is not affiliated with the upstream projects above.

## Responsible use

Voice cloning should only be performed with recordings the operator has the right and permission to use.

V-Lab is designed as a local experimentation and development environment. The person operating it is responsible for how cloned or generated speech is created and used.
