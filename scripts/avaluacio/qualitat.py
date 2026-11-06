#!/usr/bin/env python3
"""Qualitat i intel·ligibilitat: PESQ i STOI estimats sense referència (TorchAudio-Squim) i WER.

El WER es calcula amb un reconeixedor compatible amb l'API d'OpenAI (/v1/audio/transcriptions);
per defecte, el Whisper del DeepLab de la Universitat Jaume I. Es pot canviar amb STT_URL i STT_MODEL.
Abans de comparar es normalitzen variants valencianes i centrals (hui/avui, este/aquest, huit/vuit).

  .venv-voz/bin/python scripts/avaluacio/qualitat.py avl-v4-e25=tts/runs/avl-v4/mostres-sonores/epoch=0025:7
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import uuid
from pathlib import Path

import librosa
import numpy as np
import torch

STT_URL = os.environ.get("STT_URL", "https://krono.act.uji.es/deeplab/speech/v1/audio/transcriptions")
STT_MODEL = os.environ.get("STT_MODEL", "whisper-1")
EQUIV = {"avui": "hui", "aquest": "este", "vuit": "huit"}


def normalitza(t: str) -> list[str]:
    t = re.sub(r"[^\w' ]", " ", t.lower().replace("’", "'")).replace("'", " ")
    return [EQUIV.get(w, w) for w in t.split()]


def distancia(r: list[str], h: list[str]) -> int:
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1]


def transcriu(path: Path) -> str:
    b = uuid.uuid4().hex
    camps = f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{STT_MODEL}\r\n" \
            f"--{b}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nca\r\n"
    body = (camps + f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(STT_URL, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    for intent in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["text"]
        except Exception:
            time.sleep(3 * (intent + 1))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frases", type=Path, default=Path(__file__).resolve().parents[1] / "frases-sonores.json")
    ap.add_argument("veus", nargs="+", metavar="NOM=CARPETA:SPK")
    args = ap.parse_args()

    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model()
    frases = json.loads(args.frases.read_text(encoding="utf-8"))
    for v in args.veus:
        nom, resta = v.split("=", 1)
        carpeta, spk = resta.rsplit(":", 1)
        pesq, stoi, errors, paraules = [], [], 0, 0
        for i, text in enumerate(frases, 1):
            wav = Path(carpeta) / f"frase_{i:02d}_spk{spk}.wav"
            x, _ = librosa.load(wav, sr=16000)
            with torch.no_grad():
                s, p, _ = model(torch.from_numpy(x)[None])
            stoi.append(float(s))
            pesq.append(float(p))
            ref = normalitza(text)
            errors += distancia(ref, normalitza(transcriu(wav)))
            paraules += len(ref)
        print(f"{nom:24} PESQ~{np.mean(pesq):.2f}  STOI~{np.mean(stoi):.3f}  WER {100 * errors / paraules:4.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
