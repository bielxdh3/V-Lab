from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

ROOT = Path(r"C:\IA DE VOZ")
QWEN = ROOT / "qwen3-tts"
sys.path.insert(0, str(QWEN / "finetuning"))
from dataset import TTSDataset  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


def train() -> None:
    parser = argparse.ArgumentParser(description="Runner local do SFT oficial Qwen3-TTS")
    parser.add_argument("--init_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", default="speaker_1")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA é obrigatória para o SFT deste runner")
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.float16 if capability[0] < 8 else torch.bfloat16
    mixed = "fp16" if dtype == torch.float16 else "bf16"
    accelerator = Accelerator(gradient_accumulation_steps=max(1, args.gradient_accumulation), mixed_precision=mixed)

    # GTX 1060 and older cards must use eager attention; FlashAttention/BF16 are optional.
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=dtype, attn_implementation="eager")
    if hasattr(qwen3tts.model, "gradient_checkpointing_enable"):
        qwen3tts.model.gradient_checkpointing_enable()
    config = AutoConfig.from_pretrained(args.init_model_path)
    data = [json.loads(line) for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not data:
        raise ValueError("dataset vazio")
    dataset = TTSDataset(data, qwen3tts.processor, config)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=True, collate_fn=dataset.collate_fn)
    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, loader = accelerator.prepare(qwen3tts.model, optimizer, loader)
    target_speaker_embedding = None
    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding
                input_ids = batch["input_ids"]; codec_ids = batch["codec_ids"]
                text_mask = batch["text_embedding_mask"]; codec_mask_emb = batch["codec_embedding_mask"]
                attention = batch["attention_mask"]; labels = batch["codec_0_labels"]; codec_mask = batch["codec_mask"]
                text_emb = model.talker.model.text_embedding(input_ids[:, :, 0]) * text_mask
                codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_mask_emb
                codec_emb[:, 6, :] = speaker_embedding
                embeddings = text_emb + codec_emb
                for i in range(1, 16):
                    embeddings = embeddings + model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)
                outputs = model.talker(inputs_embeds=embeddings[:, :-1, :], attention_mask=attention[:, :-1], labels=labels[:, 1:], output_hidden_states=True)
                hidden = outputs.hidden_states[0][-1]
                sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(hidden[codec_mask[:, :-1]], codec_ids[codec_mask])
                loss = outputs.loss + 0.3 * sub_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.print(f"epoch={epoch+1} step={step} loss={loss.item():.5f} lr={args.lr}")
        if accelerator.is_main_process:
            out = Path(args.output_model_path) / f"checkpoint-epoch-{epoch+1}"
            shutil.copytree(args.init_model_path, out, dirs_exist_ok=True)
            cfg_path = out / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["tts_model_type"] = "custom_voice"
            tc = cfg.setdefault("talker_config", {})
            tc["spk_id"] = {args.speaker_name: 3000}; tc["spk_is_dialect"] = {args.speaker_name: False}
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            state = {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items() if not k.startswith("speaker_encoder")}
            weight = state["talker.model.codec_embedding.weight"]
            weight[3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)
            save_file(state, str(out / "model.safetensors"))
            accelerator.print(f"checkpoint={out}")


if __name__ == "__main__":
    train()
