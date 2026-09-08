"""CLI para probar un checkpoint de CompressedLLM."""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

# Permite ejecutar: python inference.py desde LLM/compressed-llm
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compressed_llm.config import Config
from model.compressed_llm import CompressedLLM
from tokenizer.tokenizer import BPETokenizer
from inference.generate import generate_texts


def load_config(checkpoint_dir: str) -> Config:
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No existe {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.from_dict(data)


def load_tokenizer(root: str) -> BPETokenizer:
    path = os.path.join(root, "tokenizer", "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el tokenizer entrenado: {path}")
    return BPETokenizer.load(path)


def encode(tokenizer, text: str, device: torch.device):
    ids = tokenizer.encode(text, add_special_tokens=True)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool, device=device)
    return ids, mask


def main():
    parser = argparse.ArgumentParser(description="Genera texto con un checkpoint de CompressedLLM")
    parser.add_argument("--checkpoint", default="checkpoints/v0", help="Directorio que contiene checkpoint.pt y config.yaml")
    parser.add_argument("--prompt", default=None, help="Prompt a probar")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Dispositivo")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Cargando checkpoint: {checkpoint_dir}")
    print(f"Dispositivo: {device}")

    cfg = load_config(checkpoint_dir)
    tokenizer = load_tokenizer(ROOT)

    # El checkpoint contiene la configuración real usada durante entrenamiento.
    payload = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    use_compression = payload.get("use_compression", True)
    model = CompressedLLM(cfg, use_compression=use_compression)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Compresión: {'ON' if use_compression else 'OFF'}")
    if use_compression:
        print(f"Latents: {cfg.compressor.latent_count}")

    prompt = args.prompt
    if prompt is None:
        print("\nEscribe un prompt. Ctrl+C para salir.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            run_one(model, tokenizer, prompt, cfg, args, device)
    else:
        run_one(model, tokenizer, prompt, cfg, args, device)


def run_one(model, tokenizer, prompt, cfg, args, device):
    context_ids, context_mask = encode(tokenizer, prompt, device)
    input_tokens = int(context_mask.sum().item())

    if model.use_compression:
        with torch.no_grad():
            latents = model.compress_context(context_ids, context_mask)
        latent_tokens = latents.size(1)
        ratio = input_tokens / latent_tokens if latent_tokens else 0
    else:
        latent_tokens = input_tokens
        ratio = 1.0

    texts = generate_texts(
        model,
        context_ids,
        context_mask,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        temperature=max(args.temperature, 1e-5),
    )

    print(f"\n[input tokens: {input_tokens} | latents: {latent_tokens} | compresión: {ratio:.2f}x]")
    print(f"Respuesta: {texts[0]}\n")


if __name__ == "__main__":
    main()
