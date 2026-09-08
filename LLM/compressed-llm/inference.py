"""CLI para probar un checkpoint de CompressedLLM."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compressed_llm.config import Config
from model.compressed_llm import CompressedLLM
from tokenizer.tokenizer import BPETokenizer
from inference.generate import generate_texts


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def resolve_checkpoint(path: str) -> str:
    """Resuelve un checkpoint explícito o detecta el último step_*.pt."""
    candidate = os.path.abspath(path)

    if os.path.isfile(candidate):
        return candidate

    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"No existe el checkpoint o directorio: {candidate}")

    checkpoints = glob.glob(os.path.join(candidate, "step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No se encontraron checkpoints step_*.pt en: {candidate}"
        )

    def step_number(checkpoint: str) -> int:
        match = _STEP_RE.search(os.path.basename(checkpoint))
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=step_number)


def load_checkpoint(path: str):
    checkpoint_path = resolve_checkpoint(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise KeyError("El checkpoint no contiene la configuración 'config'.")
    if "model_state" not in payload:
        raise KeyError("El checkpoint no contiene 'model_state'.")
    cfg = Config.from_dict(payload["config"])
    return checkpoint_path, payload, cfg


def load_tokenizer(root: str) -> BPETokenizer:
    path = os.path.join(root, "tokenizer", "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el tokenizer entrenado: {path}")
    tok = BPETokenizer()
    tok.load(path)
    return tok


def encode(tokenizer, text: str, device: torch.device):
    ids = tokenizer.encode(text, add_special=True)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool, device=device)
    return ids, mask


def main():
    parser = argparse.ArgumentParser(
        description="Genera texto con un checkpoint de CompressedLLM"
    )
    parser.add_argument(
        "--checkpoint",
        default="results/v0",
        help="Checkpoint .pt o directorio con step_*.pt (por defecto: results/v0)",
    )
    parser.add_argument("--prompt", default=None, help="Prompt a probar")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"], help="Dispositivo"
    )
    args = parser.parse_args()

    checkpoint_arg = os.path.abspath(args.checkpoint)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Buscando checkpoint en: {checkpoint_arg}")
    print(f"Dispositivo: {device}")

    checkpoint_path, payload, cfg = load_checkpoint(checkpoint_arg)
    tokenizer = load_tokenizer(ROOT)
    use_compression = payload.get("use_compression", True)
    model = CompressedLLM(cfg, use_compression=use_compression)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    print(f"Checkpoint seleccionado: {checkpoint_path}")
    print(f"Step: {payload.get('step', '?')}")
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Compresión: {'ON' if use_compression else 'OFF'}")
    if use_compression:
        print(f"Latents: {cfg.compressor.latent_count}")

    if args.prompt is None:
        print("\nEscribe un prompt. Ctrl+C para salir.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt:
                run_one(model, tokenizer, prompt, cfg, args, device)
    else:
        run_one(model, tokenizer, args.prompt, cfg, args, device)


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
