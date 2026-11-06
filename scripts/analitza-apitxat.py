#!/usr/bin/env python3
"""Mesura l'apitxat: si les sibil·lants sonores (z, dz, ʒ, dʒ) sonen sonores o sordes.

Per a cada WAV d'una oració de scripts/frases-sonores.json:
  1. l'API fonètica del DeepLab (align=true) situa cada fonema de la referència en el temps;
  2. al voltant de cada sibil·lant objectiu es busca la zona de fricció (energia 2,5-9 kHz);
  3. «caiguda» = quant baixa l'energia greu (<400 Hz, la barra de sonoritat) en la fricció
     respecte al màxim de les vocals veïnes. Una sonora la conserva (caiguda xicoteta, −5…−18 dB),
     una sorda la perd (−25…−40 dB).
El «contrast» d'una veu és la caiguda mitjana de les sordes menys la de les sonores, en positiu:
els locutors del DNV fan uns 15-20 dB; una veu apitxada, prop de 0.

(PhoneticXeus no servix per a jutjar-ho: diu «z» al 99 % encara que no hi haja sonoritat.)

  .venv-voz/bin/python scripts/analitza-apitxat.py --frases scripts/frases-sonores.json \
      nom=carpeta-amb-frase_NN_spkS.wav:spk ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import librosa
import numpy as np

API = os.environ.get("APITXAT_ALINEADOR", "https://krono.act.uji.es/deeplab/api/transcribe")

# Per oració: (paraula, ordre de la sibil·lant dins la paraula, és sonora?).
# Es deixen fora les finals en contacte amb vocal o consonant sonora (sandhi: «roses al» → [z]).
OBJECTIUS = [
    [("casa", 0, True), ("cosí", 0, True), ("església", 0, True), ("església", 1, True), ("costat", 0, False)],
    [("dotze", 0, True), ("roses", 0, True), ("jardí", 0, True), ("onze", 0, True), ("rosses", 0, False)],
    [("jove", 0, True), ("menja", 0, True), ("formatge", 0, True), ("platja", 0, True)],
    [("xiquet", 0, False), ("deixa", 0, False), ("cotxe", 0, False), ("caixa", 0, False)],
    [("zones", 0, True), ("presó", 0, True), ("són", 0, False), ("grises", 0, True), ("sessió", 0, False),
     ("sessió", 1, False), ("seriosa", 0, False), ("seriosa", 1, True)],
]
SIBIL = set("szʃʒ")


def transcriu(wav: Path, text: str) -> dict:
    url = API + "?" + urllib.parse.urlencode({"align": "true", "top_k": 3, "reference_text": text})
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + wav.read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    for intent in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception:
            if intent == 3:
                raise
    raise RuntimeError


def caiguda(y: np.ndarray, sr: int, t0: float, t1: float) -> tuple[float, float]:
    """(caiguda en dB, durada de la fricció en ms) al voltant de [t0, t1]."""
    a, b = max(0, int((t0 - 0.1) * sr)), min(len(y), int((t1 + 0.1) * sr))
    hop = int(0.005 * sr)
    S = np.abs(librosa.stft(y[a:b], n_fft=1024, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf = S[(f > 2500) & (f < 9000)].sum(0)
    lf = S[(f > 60) & (f < 400)].sum(0)
    mid = S[(f > 400) & (f < 2500)].sum(0)
    t = np.arange(len(hf)) * hop / sr + a / sr
    k = int(np.argmax(hf * np.exp(-((t - (t0 + t1) / 2) / 0.06) ** 2)))
    i = j = k
    while i > 0 and hf[i - 1] > hf[k] * 0.3:
        i -= 1
    while j < len(hf) - 1 and hf[j + 1] > hf[k] * 0.3:
        j += 1
    zona = [n for n in range(i, j + 1) if hf[n] > mid[n] * 0.5] or list(range(i, j + 1))
    return float(10 * np.log10((np.median(lf[zona]) + 1e-12) / (lf.max() + 1e-12))), len(zona) * 5.0


def temps(ops: list[dict], k: int) -> tuple[float, float] | None:
    o = ops[k]
    if o.get("start_s") is not None:
        return o["start_s"], o["end_s"]
    abans = next((x["end_s"] for x in reversed(ops[:k]) if x.get("end_s") is not None), None)
    despres = next((x["start_s"] for x in ops[k + 1:] if x.get("start_s") is not None), None)
    return (abans, despres) if abans is not None and despres is not None else None


def analitza(wav: Path, text: str, objectius: list) -> list[dict]:
    r = transcriu(wav, text)
    al = r["alignment"]
    paraules = [w["word"].lower().strip(".,;:!?«»").split("'")[-1] for w in al["reference"]]
    y, sr = librosa.load(wav, sr=None)
    out = []
    usades: dict[str, int] = {}
    for paraula, ordre, sonora in objectius:
        # paraules repetides: la n-èsima aparició
        vist = usades.get(f"{paraula}{ordre}", 0)
        idx = [i for i, w in enumerate(paraules) if w == paraula]
        if not idx:
            out.append({"paraula": paraula, "ordre": ordre, "sonora": sonora, "error": "no alineada"})
            continue
        wi = idx[min(vist, len(idx) - 1)]
        ks = [k for k, o in enumerate(al["ops"]) if o.get("word") == wi and o.get("ref") and SIBIL & set(o["ref"])]
        # «dʒ» pot vindre com a d + ʒ: cada sibil·lant és una sola entrada
        if len(ks) <= ordre or (tt := temps(al["ops"], ks[ordre])) is None:
            out.append({"paraula": paraula, "ordre": ordre, "sonora": sonora, "error": "sense temps"})
            continue
        o = al["ops"][ks[ordre]]
        c, dur = caiguda(y, sr, *tt)
        out.append({"paraula": paraula, "ordre": ordre, "sonora": sonora, "ref": o["ref"], "oit": o.get("hyp"),
                    "caiguda": round(c, 1), "dur_ms": dur})
    return out


def resum(files: list[dict]) -> dict:
    son = [f["caiguda"] for f in files if "caiguda" in f and f["sonora"]]
    sor = [f["caiguda"] for f in files if "caiguda" in f and not f["sonora"]]
    return {"sonores": round(float(np.mean(son)), 1) if son else None,
            "sordes": round(float(np.mean(sor)), 1) if sor else None,
            "contrast": round(float(np.mean(son) - np.mean(sor)), 1) if son and sor else None,
            "n": len(son) + len(sor)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frases", type=Path, default=Path(__file__).with_name("frases-sonores.json"))
    ap.add_argument("--eixida", type=Path, help="JSON amb tots els resultats")
    ap.add_argument("veus", nargs="+", metavar="NOM=CARPETA:SPK")
    args = ap.parse_args()
    frases = json.loads(args.frases.read_text(encoding="utf-8"))
    resultats = {}
    for v in args.veus:
        nom, resta = v.split("=", 1)
        carpeta, spk = resta.rsplit(":", 1)
        files = []
        for i, (text, obj) in enumerate(zip(frases, OBJECTIUS), 1):
            wav = Path(carpeta) / f"frase_{i:02d}_spk{spk}.wav"
            if not wav.exists():
                continue
            for f in analitza(wav, text, obj):
                files.append({"frase": i, **f})
        resultats[nom] = {"resum": resum(files), "files": files}
        r = resultats[nom]["resum"]
        print(f"{nom:28} sonores {r['sonores']:>6} dB  sordes {r['sordes']:>6} dB  contrast {r['contrast']:>5} dB  (n={r['n']})",
              flush=True)
    if args.eixida:
        args.eixida.write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
    # detall per paraula
    noms = list(resultats)
    print("\n" + f"{'paraula':16}" + "".join(f"{n[:14]:>15}" for n in noms))
    claus = [(f["frase"], f["paraula"], f["ordre"], f["sonora"]) for f in resultats[noms[0]]["files"]]
    for fr, p, o, s in claus:
        fila = f"{('+' if s else '-') + p + (str(o) if o else ''):16}"
        for n in noms:
            m = next((f for f in resultats[n]["files"] if (f["frase"], f["paraula"], f["ordre"]) == (fr, p, o)), {})
            fila += f"{m.get('caiguda', '—'):>15}"
        print(fila)


if __name__ == "__main__":
    sys.exit(main())
