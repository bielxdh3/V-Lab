# IA DE VOZ — Qwen3-TTS local

1. Coloque os áudios autorizados em `C:\IA DE VOZ\input_audio`.
2. Abra `C:\IA DE VOZ\INICIAR.bat`.
3. Clique **PREPARAR DATASET** e aguarde ASR, filtragem, referência e tokenizer.
4. Revise a tabela, edite transcrições/classificações se quiser e reconstrua.
5. Em uma GPU compatível, clique **INICIAR TREINAMENTO**.
6. Compare checkpoints e gere texto na aba **TESTAR VOZ**.

O alvo padrão é `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; o tokenizer e o SFT são os scripts oficiais do repositório clonado em `qwen3-tts`. O modo automático usa FP16/eager em GPUs anteriores a Ampere, batch 1 e acumulação de gradiente. FlashAttention 2 não é obrigatório.

Antes de um treino, o sistema cria um baseline zero-shot de frases fixas em `generated\baseline`. Cada geração também recebe um `.json` com WER/CER, confiança ASR e uma similaridade acústica indicativa.

Todos os caches, modelos, logs, datasets e saídas ficam dentro de `C:\IA DE VOZ`. Os arquivos em `input_audio` nunca são alterados; rejeitados são copiados para `dataset\rejected`.

Problemas conhecidos:

- A GTX 1060 6 GB normalmente não comporta o SFT 1.7B oficial; o pre-flight bloqueia o treino em vez de fingir sucesso. A inferência zero-shot pode ainda caber, dependendo da memória livre.
- O primeiro uso baixa o modelo ASR e os modelos Hugging Face e pode levar alguns minutos.
- LoRA/PEFT aparece como experimental, mas não é habilitado automaticamente sem validação do fluxo Qwen nesta máquina.
- O SFT 0.6B é bloqueado por uma incompatibilidade de dimensões no `sft_12hz.py` oficial atual; o 0.6B foi validado para inferência local em CPU.
- Diarização robusta de múltiplos falantes não é inferida por heurística: trechos com dúvida devem ficar em **REVISAR** e ser conferidos na aba de revisão antes do treino.
- Produção e revisão de qualidade continuam sendo decisões do operador; confiança ASR e speaker similarity são apenas apoio.
