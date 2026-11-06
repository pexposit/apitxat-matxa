#!/usr/bin/env python3
"""Prosòdia: mediana d'F0 i variació d'F0 (desviació típica en semitons respecte a la mediana de cada frase).

Una veu monòtona (variació baixa) sona robòtica. F0 amb pYIN (librosa), 80-450 Hz.

  .venv-voz/bin/python scripts/avaluacio/prosodia.py avl-v4-e25=tts/runs/avl-v4/mostres-sonores/epoch=0025:7
"""
from __future__ import annotations

import argparse
import glob

import librosa
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("veus", nargs="+", metavar="NOM=CARPETA:SPK")
    args = ap.parse_args()
    for v in args.veus:
        nom, resta = v.split("=", 1)
        carpeta, spk = resta.rsplit(":", 1)
        variacions, medianes = [], []
        for f in sorted(glob.glob(f"{carpeta}/frase_*_spk{spk}.wav")):
            y, sr = librosa.load(f, sr=22050)
            f0, sonor, _ = librosa.pyin(y, fmin=80, fmax=450, sr=sr, frame_length=1024)
            f0 = f0[sonor & ~np.isnan(f0)]
            st = 12 * np.log2(f0 / np.median(f0))
            variacions.append(np.std(st))
            medianes.append(np.median(f0))
        print(f"{nom:24} F0 {np.mean(medianes):5.0f} Hz  variació {np.mean(variacions):.2f} st", flush=True)


if __name__ == "__main__":
    main()
