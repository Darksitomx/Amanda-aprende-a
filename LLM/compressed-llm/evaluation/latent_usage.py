"""Pruebas de utilización de latents (§29, §30).

Instrumenta los latents para detectar colapso del bottleneck:
- norma media por latent;
- varianza por dimensión;
- similitud coseno entre latents;
- sensibilidad de la respuesta al enmascaramiento de un latent (z_i = 0).

La "prueba de enmascaramiento" (§30) oculta uno o varios latents durante
inferencia y mide el cambio en la salida/loss: latents muertos o redundantes
provocan cambios mínimos.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def latent_stats(latents: torch.Tensor) -> dict:
    """latents: [B, K, D]. Devuelve métricas agregadas de colapso."""
    k = latents.size(1)
    norms = latents.norm(dim=-1)                       # [B, K]
    mean_norm_per_latent = norms.mean(dim=0)           # [K]
    var_per_dim = latents.var(dim=1).mean(dim=0)       # [D]

    # similitud coseno media entre pares de latents (pool batch)
    flat = latents.reshape(-1, latents.size(-1))       # [B*K, D]
    flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
    cos = flat @ flat.t()                              # [B*K, B*K]
    n = cos.size(0)
    off_diag = cos - torch.eye(n, device=cos.device)
    avg_cos = off_diag.abs().mean().item()

    return {
        "latent_count": k,
        "mean_norm_per_latent": mean_norm_per_latent.tolist(),
        "mean_var_per_dim": var_per_dim.mean().item(),
        "mean_abs_cosine_pairwise": avg_cos,
        "norm_std": norms.mean(dim=1).std().item(),
    }


@torch.no_grad()
def mask_sensitivity(model, context_ids, context_mask,
                     output_ids, labels, mask_indices=None) -> dict:
    """Mide cómo cambia la loss al enmascarar cada latent (z_i -> 0).

    Returns
    -------
        ``{"baseline_loss": float, "per_index": {i: loss}}``.
    """
    model.compressor.eval()
    with torch.no_grad():
        base_out = model.forward(context_ids, context_mask,
                                 output_ids=output_ids, labels=labels)
        base_loss = base_out["loss"].item()
        K = model.cfg.compressor.latent_count
        per = {}
        if mask_indices is None:
            mask_indices = list(range(K))
        for i in mask_indices:
            lat = model.compress_context(context_ids, context_mask)
            lat = lat.clone()
            lat[:, i, :] = 0.0
            # re-insertar en el forward (monkey-patch equivalente):
            # usamos compress_context y luego ensamblamos manualmente
            out_hidden = model.tok_emb(output_ids)
            seq = torch.cat([lat, out_hidden], dim=1)
            mask = torch.cat([
                torch.ones(lat.size(0), lat.size(1), dtype=torch.bool,
                           device=lat.device),
                torch.ones(out_hidden.size(0), out_hidden.size(1),
                           dtype=torch.bool, device=lat.device),
            ], dim=1)
            logits = model.decoder(seq, mask)
            L = labels.size(1)
            resp = logits[:, lat.size(1) - 1: lat.size(1) - 1 + L, :]
            loss = torch.nn.functional.cross_entropy(
                resp.reshape(-1, resp.size(-1)), labels.reshape(-1),
                ignore_index=-100)
            per[i] = loss.item()
    model.compressor.train()
    return {"baseline_loss": base_loss, "per_index": per}