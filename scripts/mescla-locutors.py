#!/usr/bin/env python3
"""Sintetitza amb una barreja de dos locutors de Matxa: emb = (1−α)·A + α·B.

Servix per a vore quant del locutor B cal per a heretar-ne un tret (p. ex. les sibil·lants
sonores de la dona del DNV) sense perdre el timbre d'A (gina). Els WAV ixen en
<eixida>/alfa=0.30/frase_NN_spkA.wav, amb llavor fixa.

  CUDA_VISIBLE_DEVICES=1 .venv-voz/bin/python scripts/mescla-locutors.py \
      --ckpt tts/runs/avl-v1/checkpoints/epoch=0029.ckpt --a 7 --b 5 --alfa 0 0.2 0.4 \
      --frases scripts/frases-sonores.json --eixida tts/proves/mescla-gina-dona
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matxa_entorn as mx  # noqa: E402  (abans de qualsevol import de matcha)

escriu = __import__("sintetitza-matxa").escriu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--a", type=int, default=7)
    ap.add_argument("--b", type=int, default=5)
    ap.add_argument("--alfa", type=float, nargs="+", required=True)
    ap.add_argument("--frases", type=Path, default=Path(__file__).with_name("frases-sonores.json"))
    ap.add_argument("--eixida", type=Path, required=True)
    ap.add_argument("--fonemitzador", choices=["espeak", "avl"], default="avl")
    ap.add_argument("--llavor", type=int, default=1234)
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frases = json.loads(args.frases.read_text(encoding="utf-8"))
    model = mx.carrega_model(args.ckpt, device)
    voc = mx.carrega_vocoder(device, mx.vocoder_del_run(args.ckpt))
    pesos = model.spk_emb.weight
    a, b = pesos[args.a].detach().clone(), pesos[args.b].detach().clone()
    for alfa in args.alfa:
        with torch.no_grad():
            pesos[args.a].copy_((1 - alfa) * a + alfa * b)
        eixida = args.eixida / f"alfa={alfa:.2f}"
        eixida.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(frases, 1):
            torch.manual_seed(args.llavor)
            escriu(eixida / f"frase_{i:02d}_spk{args.a}.wav",
                   mx.sintetitza(model, voc, text, args.a, device, fonemitzador=args.fonemitzador))
        print(f"alfa={alfa:.2f}: {len(frases)} frases", flush=True)


if __name__ == "__main__":
    main()
