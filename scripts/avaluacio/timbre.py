#!/usr/bin/env python3
"""Similitud de locutor (ECAPA-TDNN de SpeechBrain) amb una veu de referència.

Per a cada carpeta es fa la mitjana dels embeddings de les seues frases (frase_NN_spkS.wav),
es normalitza i es compara amb la mitjana de la referència pel cosinus.

  .venv-voz/bin/python scripts/avaluacio/timbre.py --ref proves/gina-original:7 \
      --ref2 proves/dona-dnv:5 avl-v4-e25=tts/runs/avl-v4/mostres-sonores/epoch=0025:7
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import librosa
import numpy as np
import torch

ARREL = Path(__file__).resolve().parents[2]


def carpeta_spk(arg: str) -> tuple[str, str]:
    d, spk = arg.rsplit(":", 1)
    return d, spk


def embedding(enc, carpeta: str, spk: str) -> np.ndarray:
    es = []
    for f in sorted(glob.glob(f"{carpeta}/frase_*_spk{spk}.wav")):
        x, _ = librosa.load(f, sr=16000)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(x).float()[None]).squeeze().numpy()
        es.append(e / np.linalg.norm(e))
    if not es:
        raise SystemExit(f"cap frase_*_spk{spk}.wav en {carpeta}")
    m = np.mean(es, 0)
    return m / np.linalg.norm(m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, metavar="CARPETA:SPK", help="veu original (p. ex. gina)")
    ap.add_argument("--ref2", metavar="CARPETA:SPK", help="segona referència opcional (p. ex. la dona del DNV)")
    ap.add_argument("veus", nargs="+", metavar="NOM=CARPETA:SPK")
    args = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=str(ARREL / "tts" / "models" / "spkrec-ecapa-voxceleb"),
                                         run_opts={"device": "cpu"})
    ref = embedding(enc, *carpeta_spk(args.ref))
    ref2 = embedding(enc, *carpeta_spk(args.ref2)) if args.ref2 else None
    for v in args.veus:
        nom, resta = v.split("=", 1)
        e = embedding(enc, *carpeta_spk(resta))
        linia = f"{nom:24} sim_ref {e @ ref:.3f}"
        if ref2 is not None:
            linia += f"   sim_ref2 {e @ ref2:.3f}"
        print(linia, flush=True)


if __name__ == "__main__":
    main()
