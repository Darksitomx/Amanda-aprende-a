"""CLI definitivo de Compressed LLM.

Un único punto de entrada para todo el ciclo de vida de un experimento:

    train     -> entrenar un modelo NUEVO desde cero (step 0)
    continue  -> seguir entrenando un experimento existente (mismo tamaño)
    grow      -> clonar un experimento a una arquitectura MÁS GRANDE,
                 transplantando los pesos ya entrenados (en vez de tirarlos)
    infer     -> generar texto con el checkpoint de un experimento
    status    -> ver el estado (steps, loss, params) de todos los experimentos

Cada experimento vive en ``results/<experiment>/`` con sus checkpoints
(``step_N.pt``), ``metrics.json`` (historial completo, no se pisa entre
llamadas) y ``config_resolved.json`` (la config exacta con la que se creó,
para poder reconstruir el modelo sin adivinar nada).

El dataset (``data/out/*.jsonl``) y el tokenizer (``tokenizer/tokenizer.json``)
son GLOBALES y compartidos entre experimentos: así ``grow`` puede transplantar
pesos sin romper el embedding (mismo vocabulario siempre). Si quieres tokenizer
distinto, es un proyecto nuevo, no un `grow`.

Ejemplos
--------
Entrenar desde cero (descarga datos si hacen falta, entrena tokenizer si no
existe, hace split y entrena):

    python cli.py train --experiment v0 --config v0 --rows 3000 --steps 5000

Seguir entrenando el mismo experimento más pasos (y opcionalmente más datos):

    python cli.py continue --experiment v0 --steps 5000
    python cli.py continue --experiment v0 --steps 3000 --rows 8000

Hacer crecer v0 a una arquitectura más grande definida en configs/v1.yaml,
reusando lo aprendido, y seguir entrenando 4000 steps más:

    python cli.py grow --experiment v0 --to v1 --config v1 --steps 4000

Probar el modelo:

    python cli.py infer --experiment v1
    python cli.py infer --experiment v1 --prompt "Hola, ¿cómo estás?"

Ver todos los experimentos:

    python cli.py status

Sin argumentos abre un menú interactivo equivalente a todo lo anterior.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_ROOT = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
OUT_DIR = ROOT / "data" / "out"

# Helpers de datos/tokenizer/dispositivo ya existentes y probados en
# train_cli.py; se reusan tal cual para no duplicar lógica. Importar este
# módulo no ejecuta nada (su main() está bajo `if __name__ == "__main__"`).
from train_cli import (  # noqa: E402
    fetch_more_rows,
    preprocess_all,
    ensure_tokenizer,
    build_splits,
    detect_device,
    recommended_batch_size,
    print_gpu_info,
    existing_raw_rows,
    make_loaders,
)
from train import run_tests, run_overfit  # noqa: E402

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


# --------------------------------------------------------------------------- #
# Utilidades de experimento                                                    #
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def experiment_dir(name: str) -> Path:
    return RESULTS_ROOT / name


def latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    if not exp_dir.exists():
        return None
    candidates = []
    for path in exp_dir.glob("step_*.pt"):
        match = _STEP_RE.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def checkpoint_step(path: Optional[Path]) -> int:
    if path is None:
        return 0
    match = _STEP_RE.search(path.name)
    return int(match.group(1)) if match else 0


def load_config_yaml(name: str):
    from compressed_llm.config import Config
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe configs/{name}.yaml")
    return Config.from_yaml(str(path))


def save_experiment_config(exp_dir: Path, cfg) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config_resolved.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_experiment_config(exp_dir: Path):
    from compressed_llm.config import Config
    path = exp_dir / "config_resolved.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe. ¿Es un experimento válido creado con 'train' o 'grow'?"
        )
    return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_tokenizer():
    from tokenizer.tokenizer import BPETokenizer
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"No existe {TOKENIZER_PATH}. Entrena un experimento con 'train' primero "
            "(eso crea el tokenizer una única vez para todo el proyecto)."
        )
    tok = BPETokenizer()
    tok.load(str(TOKENIZER_PATH))
    return tok


def save_run_summary(exp_dir: Path, extra: Dict) -> None:
    path = exp_dir / "run_summary.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(extra)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_data(rows: int, tokenizer_vocab: int):
    """Descarga/acumula filas, preprocesa, asegura tokenizer y arma splits.

    Devuelve ``(tok, train_records, valid_records, test_records)``.
    """
    raw_rows = fetch_more_rows(rows)
    records = preprocess_all(raw_rows)
    tok = ensure_tokenizer(records, tokenizer_vocab)
    train_records, valid_records, test_records = build_splits(records, seed=42)
    return tok, train_records, valid_records, test_records


def apply_tokenizer_to_config(cfg, tok) -> None:
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.pad_token_id = tok.pad_token_id
    cfg.model.bos_token_id = tok.bos_token_id
    cfg.model.eos_token_id = tok.eos_token_id


def resolve_batch_accum(batch: Optional[int], accum: Optional[int],
                        effective_batch: Optional[int]) -> tuple:
    batch = batch or recommended_batch_size()
    if accum is not None:
        return batch, accum
    target_effective = effective_batch or batch * 8
    return batch, max(1, math.ceil(target_effective / batch))


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #

def cmd_train(args) -> None:
    from compressed_llm.config import Config
    from model.compressed_llm import CompressedLLM
    from training.trainer import Trainer

    exp_dir = experiment_dir(args.experiment)
    if latest_checkpoint(exp_dir) is not None and not args.fresh:
        raise SystemExit(
            f"Ya existe un experimento '{args.experiment}' con checkpoints en {exp_dir}.\n"
            f"Usa 'continue --experiment {args.experiment}' para seguir entrenándolo, "
            f"o pasa --fresh para borrarlo y empezar de cero."
        )
    if args.fresh and exp_dir.exists():
        shutil.rmtree(exp_dir)

    banner(f"TRAIN — nuevo experimento '{args.experiment}' (config: {args.config})")
    print_gpu_info()

    cfg = load_config_yaml(args.config)
    tok, train_records, valid_records, _ = prepare_data(args.rows, args.tokenizer_vocab)
    apply_tokenizer_to_config(cfg, tok)

    if args.verify:
        run_tests()
        run_overfit(cfg, tok, train_records, args.overfit_steps, args.device)

    batch, accum = resolve_batch_accum(args.batch, args.accum, args.effective_batch)
    cfg.training.batch_size = batch
    cfg.training.gradient_accumulation = accum
    device = args.device or detect_device()

    train_loader, valid_loader = make_loaders(cfg, tok, train_records, valid_records, batch)
    model = CompressedLLM(cfg, use_compression=not args.baseline)
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Micro-batch: {batch} | accumulation: {accum} | batch efectivo: {batch * accum}")
    print(f"Input tokens: {cfg.training.max_input_tokens} -> "
          f"{cfg.compressor.latent_count if not args.baseline else cfg.training.max_input_tokens} "
          f"{'latents' if not args.baseline else '(baseline, sin compresión)'}")

    save_experiment_config(exp_dir, cfg)
    trainer = Trainer(model, cfg, train_loader, valid_loader, device=device)
    result = trainer.train(max_steps=args.steps, output_dir=str(exp_dir))

    save_run_summary(exp_dir, {
        "config_name": args.config,
        "dataset_rows": existing_raw_rows(),
        "tokenizer_vocab": tok.vocab_size,
        "use_compression": not args.baseline,
        "latent_count": cfg.compressor.latent_count,
        "compression_ratio": cfg.training.max_input_tokens / max(1, cfg.compressor.latent_count),
        "parameter_count": model.get_num_parameters(),
        "batch_size": batch,
        "gradient_accumulation": accum,
        "final_step": result["final_step"],
    })
    banner(f"'{args.experiment}' entrenado hasta el step {result['final_step']}")


# --------------------------------------------------------------------------- #
# continue                                                                     #
# --------------------------------------------------------------------------- #

def cmd_continue(args) -> None:
    from model.compressed_llm import CompressedLLM
    from training.trainer import Trainer

    exp_dir = experiment_dir(args.experiment)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(exp_dir)
    if checkpoint is None:
        raise SystemExit(
            f"No hay checkpoints en {exp_dir}. Usa 'train --experiment {args.experiment}' primero."
        )
    current_step = checkpoint_step(checkpoint)
    cfg = load_experiment_config(exp_dir)

    banner(f"CONTINUE — '{args.experiment}' desde step {current_step}")
    print_gpu_info()
    print(f"Checkpoint: {checkpoint}")

    rows_target = args.rows if args.rows else existing_raw_rows()
    tok, train_records, valid_records, _ = prepare_data(rows_target, cfg.model.vocab_size)
    # El tokenizer ya existe (se creó en 'train'); no debe cambiar el
    # vocab_size a mitad de entrenamiento, así que solo lo verificamos.
    if tok.vocab_size != cfg.model.vocab_size:
        raise SystemExit(
            f"El tokenizer actual tiene vocab_size={tok.vocab_size} pero el experimento "
            f"'{args.experiment}' se creó con vocab_size={cfg.model.vocab_size}. "
            "No se puede continuar sin invalidar los embeddings ya entrenados."
        )

    batch, accum = resolve_batch_accum(
        args.batch or cfg.training.batch_size, args.accum, args.effective_batch
    )
    cfg.training.batch_size = batch
    cfg.training.gradient_accumulation = accum
    device = args.device or detect_device()

    train_loader, valid_loader = make_loaders(cfg, tok, train_records, valid_records, batch)
    use_compression = torch.load(checkpoint, map_location="cpu", weights_only=False).get(
        "use_compression", True
    )
    model = CompressedLLM(cfg, use_compression=use_compression)
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Micro-batch: {batch} | accumulation: {accum} | batch efectivo: {batch * accum}")

    target_step = args.to_step if args.to_step else current_step + args.steps
    if target_step <= current_step:
        raise SystemExit(f"El objetivo (step {target_step}) no supera el step actual ({current_step}).")

    trainer = Trainer(model, cfg, train_loader, valid_loader,
                      resume_from=str(checkpoint), device=device)
    print(f"Entrenando step {current_step} -> {target_step}")
    result = trainer.train(max_steps=target_step, output_dir=str(exp_dir))

    save_run_summary(exp_dir, {
        "dataset_rows": existing_raw_rows(),
        "parameter_count": model.get_num_parameters(),
        "batch_size": batch,
        "gradient_accumulation": accum,
        "final_step": result["final_step"],
    })
    banner(f"'{args.experiment}' entrenado hasta el step {result['final_step']}")


# --------------------------------------------------------------------------- #
# grow                                                                         #
# --------------------------------------------------------------------------- #

def cmd_grow(args) -> None:
    from model.compressed_llm import CompressedLLM
    from model.grow import grow_model
    from training.trainer import Trainer

    src_dir = experiment_dir(args.experiment)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(src_dir)
    if checkpoint is None:
        raise SystemExit(f"No hay checkpoints en {src_dir}.")

    target_dir = experiment_dir(args.to)
    if latest_checkpoint(target_dir) is not None and not args.force:
        raise SystemExit(
            f"Ya existe un experimento '{args.to}' con checkpoints. "
            "Pasa --force para sobrescribirlo."
        )

    banner(f"GROW — '{args.experiment}' (step {checkpoint_step(checkpoint)}) -> "
          f"'{args.to}' (config: {args.config})")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_cfg = load_experiment_config(src_dir)
    use_compression = payload.get("use_compression", True)

    tok = load_tokenizer()
    new_cfg = load_config_yaml(args.config)
    apply_tokenizer_to_config(new_cfg, tok)
    if new_cfg.model.vocab_size != old_cfg.model.vocab_size:
        raise SystemExit(
            "'grow' no soporta cambiar el vocabulario (invalidaría los embeddings "
            "ya entrenados). Usa el mismo tokenizer/proyecto."
        )

    old_model = CompressedLLM(old_cfg, use_compression=use_compression)
    old_model.load_state_dict(payload["model_state"])
    new_model = CompressedLLM(new_cfg, use_compression=use_compression)

    report = grow_model(old_model, new_model)
    print(f"\nParámetros: {old_model.get_num_parameters():,} -> {new_model.get_num_parameters():,}")
    print(report.summary())

    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(target_dir, new_cfg)
    torch.save({
        "model_state": new_model.state_dict(),
        "step": 0,
        "config": new_cfg.to_dict(),
        "use_compression": use_compression,
    }, target_dir / "step_0.pt")
    save_run_summary(target_dir, {
        "config_name": args.config,
        "grown_from": args.experiment,
        "grown_from_step": checkpoint_step(checkpoint),
        "tokenizer_vocab": tok.vocab_size,
        "use_compression": use_compression,
        "latent_count": new_cfg.compressor.latent_count,
        "compression_ratio": new_cfg.training.max_input_tokens / max(1, new_cfg.compressor.latent_count),
        "parameter_count": new_model.get_num_parameters(),
        "final_step": 0,
    })
    banner(f"'{args.to}' creado en step 0 a partir de '{args.experiment}'")

    if args.steps:
        print(f"\nContinuando entrenamiento de '{args.to}' por {args.steps} steps...")
        cont_args = argparse.Namespace(
            experiment=args.to, checkpoint=None, rows=args.rows, steps=args.steps,
            to_step=None, batch=args.batch, accum=args.accum,
            effective_batch=args.effective_batch, device=args.device,
        )
        cmd_continue(cont_args)


# --------------------------------------------------------------------------- #
# infer                                                                        #
# --------------------------------------------------------------------------- #

def _load_model_for_inference(exp_dir: Path, checkpoint_arg: Optional[str], device: torch.device):
    from compressed_llm.config import Config
    from model.compressed_llm import CompressedLLM

    checkpoint = Path(checkpoint_arg) if checkpoint_arg else latest_checkpoint(exp_dir)
    if checkpoint is None:
        raise SystemExit(f"No hay checkpoints en {exp_dir}.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(payload["config"])
    use_compression = payload.get("use_compression", True)
    model = CompressedLLM(cfg, use_compression=use_compression)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, cfg, checkpoint, payload.get("step", "?")


def _generate_once(model, tokenizer, prompt: str, cfg, max_new_tokens: int,
                   temperature: float, device: torch.device) -> None:
    from inference.generate import generate_texts

    ids = tokenizer.encode(prompt, add_special=True)
    context_ids = torch.tensor([ids], dtype=torch.long, device=device)
    context_mask = torch.ones_like(context_ids, dtype=torch.bool)
    input_tokens = int(context_mask.sum().item())

    if model.use_compression:
        with torch.no_grad():
            latents = model.compress_context(context_ids, context_mask)
        latent_tokens = latents.size(1)
        ratio = input_tokens / latent_tokens if latent_tokens else 0.0
    else:
        latent_tokens, ratio = input_tokens, 1.0

    texts = generate_texts(
        model, context_ids, context_mask, tokenizer,
        max_new_tokens=max_new_tokens,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        temperature=max(temperature, 1e-5),
    )
    print(f"[input tokens: {input_tokens} | latents: {latent_tokens} | compresión: {ratio:.2f}x]")
    print(f"Respuesta: {texts[0]}\n")


def cmd_infer(args) -> None:
    exp_dir = experiment_dir(args.experiment)
    device = torch.device(args.device or detect_device())
    model, cfg, checkpoint, step = _load_model_for_inference(exp_dir, args.checkpoint, device)
    tokenizer = load_tokenizer()

    print(f"Experimento: {args.experiment} | Checkpoint: {checkpoint} | Step: {step}")
    print(f"Parámetros: {model.get_num_parameters():,} | "
          f"Compresión: {'ON (' + str(cfg.compressor.latent_count) + ' latents)' if model.use_compression else 'OFF'}")

    if args.prompt:
        _generate_once(model, tokenizer, args.prompt, cfg, args.max_new_tokens,
                       args.temperature, device)
        return

    print("\nEscribe un prompt. Ctrl+C para salir.\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt:
            _generate_once(model, tokenizer, prompt, cfg, args.max_new_tokens,
                           args.temperature, device)


# --------------------------------------------------------------------------- #
# status                                                                       #
# --------------------------------------------------------------------------- #

def _last_metric(exp_dir: Path) -> Optional[Dict]:
    path = exp_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[-1] if data else None
    except Exception:
        return None


def cmd_status(args) -> None:
    banner("ESTADO DE LOS EXPERIMENTOS")
    print(f"Filas Spanish acumuladas: {existing_raw_rows()}")
    print(f"Tokenizer: {'OK -> ' + str(TOKENIZER_PATH) if TOKENIZER_PATH.exists() else 'NO EXISTE'}")

    if not RESULTS_ROOT.exists():
        print("\nNo hay ningún experimento todavía. Usa 'train' para crear el primero.")
        return

    names = [args.experiment] if args.experiment else sorted(
        d.name for d in RESULTS_ROOT.iterdir() if d.is_dir()
    )
    for name in names:
        exp_dir = experiment_dir(name)
        checkpoint = latest_checkpoint(exp_dir)
        if checkpoint is None and not (exp_dir / "config_resolved.json").exists():
            continue
        step = checkpoint_step(checkpoint)
        summary_path = exp_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        last = _last_metric(exp_dir)

        print(f"\n--- {name} " + "-" * max(0, 60 - len(name)))
        print(f"  config:           {summary.get('config_name', '?')}")
        print(f"  step actual:      {step}")
        print(f"  parámetros:       {summary.get('parameter_count', '?'):,}" if isinstance(
            summary.get("parameter_count"), int) else f"  parámetros:       {summary.get('parameter_count', '?')}")
        print(f"  compresión:       {summary.get('compression_ratio', '?')}x "
              f"({summary.get('latent_count', '?')} latents)"
              if summary.get("use_compression", True) else "  compresión:       OFF (baseline)")
        print(f"  dataset (filas):  {summary.get('dataset_rows', '?')}")
        if summary.get("grown_from"):
            print(f"  creado por grow:  desde '{summary['grown_from']}' (step {summary.get('grown_from_step')})")
        if last:
            print(f"  train_loss:       {last.get('train_loss'):.4f}")
            print(f"  valid_loss:       {last.get('valid_loss'):.4f}")
            print(f"  perplexity:       {last.get('perplexity'):.2f}")
        else:
            print("  train_loss:       (sin métricas registradas todavía)")


# --------------------------------------------------------------------------- #
# argparse                                                                     #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="CLI definitivo de Compressed LLM: train / continue / grow / infer / status",
    )
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Entrenar un experimento nuevo desde cero")
    t.add_argument("--experiment", required=True, help="Nombre del experimento (results/<experiment>/)")
    t.add_argument("--config", required=True, help="Nombre del YAML en configs/ sin .yaml")
    t.add_argument("--rows", type=int, default=2048, help="Filas Spanish objetivo (acumuladas)")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--tokenizer-vocab", type=int, default=8192)
    t.add_argument("--batch", type=int, default=None)
    t.add_argument("--accum", type=int, default=None)
    t.add_argument("--effective-batch", type=int, default=None)
    t.add_argument("--device", default=None, choices=["cpu", "cuda"])
    t.add_argument("--baseline", action="store_true", help="Desactiva la compresión (sin latents)")
    t.add_argument("--fresh", action="store_true", help="Borra el experimento si ya existe")
    t.add_argument("--verify", action="store_true", help="Corre tests + overfit-check antes de entrenar")
    t.add_argument("--overfit-steps", type=int, default=150)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("continue", help="Seguir entrenando un experimento existente")
    c.add_argument("--experiment", required=True)
    c.add_argument("--steps", type=int, default=1000, help="Steps ADICIONALES a entrenar")
    c.add_argument("--to-step", type=int, default=None, help="Alternativa: step absoluto objetivo")
    c.add_argument("--rows", type=int, default=None, help="Crecer el dataset a esta cantidad de filas")
    c.add_argument("--checkpoint", default=None, help="Checkpoint explícito (por defecto: el más reciente)")
    c.add_argument("--batch", type=int, default=None)
    c.add_argument("--accum", type=int, default=None)
    c.add_argument("--effective-batch", type=int, default=None)
    c.add_argument("--device", default=None, choices=["cpu", "cuda"])
    c.set_defaults(func=cmd_continue)

    g = sub.add_parser("grow", help="Clonar un experimento a una arquitectura más grande")
    g.add_argument("--experiment", required=True, help="Experimento origen (ya entrenado)")
    g.add_argument("--to", required=True, help="Nombre del experimento destino (más grande)")
    g.add_argument("--config", required=True, help="YAML con la arquitectura más grande")
    g.add_argument("--checkpoint", default=None, help="Checkpoint origen explícito")
    g.add_argument("--force", action="store_true", help="Sobrescribe el destino si ya existe")
    g.add_argument("--steps", type=int, default=0, help="Si > 0, continúa entrenando el destino")
    g.add_argument("--rows", type=int, default=None)
    g.add_argument("--batch", type=int, default=None)
    g.add_argument("--accum", type=int, default=None)
    g.add_argument("--effective-batch", type=int, default=None)
    g.add_argument("--device", default=None, choices=["cpu", "cuda"])
    g.set_defaults(func=cmd_grow)

    i = sub.add_parser("infer", help="Generar texto con un experimento")
    i.add_argument("--experiment", required=True)
    i.add_argument("--checkpoint", default=None)
    i.add_argument("--prompt", default=None, help="Si se omite, abre un REPL interactivo")
    i.add_argument("--max-new-tokens", type=int, default=64)
    i.add_argument("--temperature", type=float, default=0.8)
    i.add_argument("--device", default=None, choices=["cpu", "cuda"])
    i.set_defaults(func=cmd_infer)

    s = sub.add_parser("status", help="Ver el estado de todos los experimentos")
    s.add_argument("--experiment", default=None, help="Limitar a un experimento")
    s.set_defaults(func=cmd_status)

    return p


# --------------------------------------------------------------------------- #
# Menú interactivo (sin argumentos)                                           #
# --------------------------------------------------------------------------- #

def _ask(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return raw or default


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def interactive_menu() -> None:
    parser = build_parser()
    while True:
        banner("COMPRESSED LLM — CLI")
        print("1) Entrenar experimento nuevo (train)")
        print("2) Continuar entrenamiento (continue)")
        print("3) Hacer crecer un experimento (grow)")
        print("4) Inferencia / chat (infer)")
        print("5) Ver estado de experimentos (status)")
        print("6) Salir")
        choice = _ask("Selecciona", "5")
        try:
            if choice == "1":
                args = parser.parse_args([
                    "train",
                    "--experiment", _ask("Nombre del experimento", "v0"),
                    "--config", _ask("Config (configs/*.yaml)", "v0"),
                    "--rows", str(_ask_int("Filas Spanish objetivo", 2000)),
                    "--steps", str(_ask_int("Steps a entrenar", 2000)),
                ])
                args.func(args)
            elif choice == "2":
                args = parser.parse_args([
                    "continue",
                    "--experiment", _ask("Nombre del experimento", "v0"),
                    "--steps", str(_ask_int("Steps adicionales", 1000)),
                    "--rows", str(_ask_int("Filas Spanish objetivo (0 = no crecer)", 0)) or "0",
                ])
                if args.rows == 0:
                    args.rows = None
                args.func(args)
            elif choice == "3":
                args = parser.parse_args([
                    "grow",
                    "--experiment", _ask("Experimento origen", "v0"),
                    "--to", _ask("Nombre del experimento destino", "v1"),
                    "--config", _ask("Config más grande (configs/*.yaml)", "v1"),
                    "--steps", str(_ask_int("Steps a entrenar tras crecer (0 = solo crear)", 2000)),
                ])
                args.func(args)
            elif choice == "4":
                args = parser.parse_args([
                    "infer",
                    "--experiment", _ask("Nombre del experimento", "v0"),
                ])
                args.func(args)
            elif choice == "5":
                args = parser.parse_args(["status"])
                args.func(args)
            else:
                print("Hasta luego.")
                return
        except KeyboardInterrupt:
            print("\nInterrumpido. El último checkpoint guardado queda intacto.")
        except SystemExit as exc:
            print(f"\n{exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"\nERROR: {type(exc).__name__}: {exc}")
        input("\nPulsa ENTER para volver al menú...")


def main() -> None:
    os.chdir(ROOT)
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        interactive_menu()
        return
    args.func(args)


if __name__ == "__main__":
    main()
