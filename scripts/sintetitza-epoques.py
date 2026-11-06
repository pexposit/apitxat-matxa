#!/usr/bin/env python3
"""Sintetitza unes frases amb cada checkpoint d'un fine-tune, per a escoltar com canvia.

A diferència de les mostres que fa l'entrenament, ací triem les frases (per defecte
scripts/frases-avl.json, plenes de paraules on la fonètica AVL no és la d'espeak) i fixem la
llavor: la mateixa frase ix amb el mateix soroll en totes les èpoques, així que les diferències
són del model i no de l'atzar. Es pot llançar mentre l'entrenament continua; només fa els
checkpoints que encara no tenen totes les frases.

  CUDA_VISIBLE_DEVICES=1 .venv-voz/bin/python scripts/sintetitza-epoques.py --run avl-v1 --spk 7 --spk 5

Els WAV ixen en tts/runs/<run>/mostres-avl/epoch=NNNN/frase_NN_spkS.wav.
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--spk", type=int, action="append", required=True)
    ap.add_argument("--frases", type=Path, default=Path(__file__).with_name("frases-avl.json"))
    ap.add_argument("--carpeta", default="mostres-avl")
    ap.add_argument("--fonemitzador", choices=["espeak", "avl"], default="avl")
    ap.add_argument("--llavor", type=int, default=1234)
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frases = json.loads(args.frases.read_text(encoding="utf-8"))
    run = mx.TTS / "runs" / args.run
    for ckpt in sorted((run / "checkpoints").glob("epoch=*.ckpt")):
        eixida = run / args.carpeta / ckpt.stem
        wavs = [(eixida / f"frase_{i:02d}_spk{spk}.wav", text, spk)
                for spk in args.spk for i, text in enumerate(frases, 1)]
        if all(w.exists() for w, _, _ in wavs):
            continue
        eixida.mkdir(parents=True, exist_ok=True)
        model = mx.carrega_model(ckpt, device)
        voc = mx.carrega_vocoder(device, mx.vocoder_del_run(ckpt))
        for wav, text, spk in wavs:
            torch.manual_seed(args.llavor)
            escriu(wav, mx.sintetitza(model, voc, text, spk, device, fonemitzador=args.fonemitzador))
        print(f"{ckpt.stem}: {len(wavs)} frases", flush=True)
        del model, voc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
