#!/usr/bin/env python3
"""Guarda un checkpoint de Matxa amb l'embedding d'un locutor barrejat amb el d'un altre.

És el que fa scripts/mescla-locutors.py en inferència, però fix: el checkpoint resultant
es pot usar (o exportar a ONNX) com qualsevol altre, i el locutor A ja hi té la barreja.

  .venv-voz/bin/python scripts/hornea-barreja.py tts/runs/avl-v2/checkpoints/epoch=0029.ckpt \
      tts/runs/avl-v2/checkpoints/epoch=0029-dona15.ckpt --a 7 --b 5 --alfa 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path)
    ap.add_argument("eixida", type=Path)
    ap.add_argument("--a", type=int, default=7)
    ap.add_argument("--b", type=int, default=5)
    ap.add_argument("--alfa", type=float, required=True)
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.entrada, map_location="cpu", weights_only=False)
    w = ckpt["state_dict"]["spk_emb.weight"]
    w[args.a] = (1 - args.alfa) * w[args.a] + args.alfa * w[args.b]
    # sense l'estat de l'optimitzador: només per a inferència
    for clau in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
        ckpt.pop(clau, None)
    torch.save(ckpt, args.eixida)
    print(f"{args.eixida}: locutor {args.a} = {1 - args.alfa:.2f}·{args.a} + {args.alfa:.2f}·{args.b}")


if __name__ == "__main__":
    main()
