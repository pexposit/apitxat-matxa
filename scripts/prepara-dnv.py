#!/usr/bin/env python3
"""Dataset de clips per a Matxa amb les gravacions del DNV i la seua transcripció AVL.

Cada gravació de descarrega-dnv.py diu una o més formes («col·legi o col·legi»,
«metge, metgessa», «poder, pot»). Es parteix pels silencis i, si hi ha tants
trams de veu com transcripcions, cada tram va amb la seua; si no, l'entrada es
descarta (es diu d'una tirada i no se sap on talla). Els clips es retallen amb
un poc de silenci a cada costat i es normalitzen de volum.

El text fonètic no el fa espeak: és la transcripció de l'AVL passada als
símbols de Matxa (lexic_avl.a_matxa), en la tercera columna de metadata.csv,
que prepara-matxa.py fa servir tal qual.

Unes paraules es reserven per a avaluar (no entren al dataset): les que tenen
discrepància amb espeak, triades amb llavor fixa (`--reserva`).

  .venv-voz/bin/python scripts/prepara-dnv.py
  .venv-voz/bin/python scripts/prepara-matxa.py --eixida voz-dataset/matxa-dnv \
      --font voz-dataset/clips-dnv/home:<spk> --font voz-dataset/clips-dnv/dona:<spk>

El DNV té dues veus, un home (~90 Hz) i una dona (~190 Hz): cada gravació va a
home/ o dona/ segons la seua f0 mitjana, perquè s'entrenen com a locutors distints.

Eixida (voz-dataset/clips-dnv/): home/ i dona/ (wavs/ i metadata.csv amb
id|paraula|fonemes), locutors.tsv (veu, f0 i durada de cada clip),
reservades.tsv (paraula, fonemes AVL, fitxer original, veu) i descartades.tsv.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexic_avl  # noqa: E402

LAB = Path(__file__).resolve().parent.parent
DNV = LAB / "voz-dataset" / "dnv"
SR = 22050
MARGE = 0.12  # segons de silenci que es deixen a cada costat de la paraula
PAUSA_MIN = 0.25  # pauses més curtes no separen dues formes («col·legi o col·legi»)
F0_LIMIT = 145  # Hz: per davall, la veu d'home; per damunt, la de dona


def trams(y: np.ndarray, top_db: float) -> list[list[int]]:
    units: list[list[int]] = []
    for a, b in librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256):
        if units and (a - units[-1][1]) / SR < PAUSA_MIN:
            units[-1][1] = b
        else:
            units.append([int(a), int(b)])
    return units


def normalitza(y: np.ndarray) -> np.ndarray:
    pic = np.max(np.abs(y)) or 1.0
    return (y / pic * 0.9).astype(np.float32)


def f0_mitjana(y: np.ndarray) -> float:
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=SR, frame_length=1024)
    veu = f0[(f0 > 60) & (f0 < 400)]
    return float(np.median(veu)) if len(veu) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrades", type=Path, default=DNV / "entrades.jsonl")
    ap.add_argument("--eixida", type=Path, default=LAB / "voz-dataset" / "clips-dnv")
    ap.add_argument("--reserva", type=int, default=300, help="paraules amb discrepància d'espeak que no s'entrenen")
    ap.add_argument("--top-db", type=float, default=35)
    args = ap.parse_args()

    entrades = [json.loads(l) for l in args.entrades.read_text(encoding="utf-8").splitlines() if l.strip()]
    entrades = [e for e in entrades if e.get("fitxer") and e.get("formes") and "/" not in e["fonetica"]
                and not e["lema"].startswith("-") and not e["lema"].endswith("-")]

    # Reservades: entrades d'una sola forma on espeak discrepa (on es vorà si el model aprén l'AVL).
    discrepancies = {l.split("\t")[0] for l in (DNV / "discrepancies-espeak.tsv").read_text(encoding="utf-8")
                     .splitlines()[1:] if l} if (DNV / "discrepancies-espeak.tsv").exists() else set()
    candidates = sorted(e["fitxer"] for e in entrades
                        if len(e["formes"]) == 1 and e["lema"].lower() in discrepancies)
    reservades = set(random.Random(1234).sample(candidates, min(args.reserva, len(candidates))))

    for veu in ("home", "dona"):
        (args.eixida / veu / "wavs").mkdir(parents=True, exist_ok=True)
    metadata: dict[str, list[str]] = {"home": [], "dona": []}
    locutors, reserva, descartades = [], [], []
    for n, e in enumerate(entrades, 1):
        formes = [(p, lexic_avl.a_matxa(ipa)) for p, ipa, _ in lexic_avl.formes_amb_paraula(e)]
        y, _ = librosa.load(DNV / "flac" / e["fitxer"], sr=SR, mono=True)
        f0 = f0_mitjana(y)
        veu = "home" if 0 < f0 < F0_LIMIT else "dona"
        if e["fitxer"] in reservades:
            reserva.append(f"{formes[0][0]}\t{formes[0][1]}\t{e['fitxer']}\t{veu}")
            continue
        t = trams(y, args.top_db)
        if len(t) != len(formes) or any(not p for p, _ in formes):
            descartades.append(f"{e['fitxer']}\t{e['fonetica']}\t{len(t)} trams")
            continue
        base = Path(e["fitxer"]).stem
        for k, ((paraula, fon), (a, b)) in enumerate(zip(formes, t)):
            marge = int(MARGE * SR)
            clip = normalitza(y[max(0, a - marge): b + marge])
            cid = base if len(formes) == 1 else f"{base}_{k + 1}"
            sf.write(args.eixida / veu / "wavs" / f"{cid}.wav", clip, SR, subtype="PCM_16")
            metadata[veu].append(f"{cid}|{paraula}|{fon}")
            locutors.append(f"{cid}\t{veu}\t{f0:.0f}\t{len(clip) / SR:.2f}")
        if n % 1000 == 0:
            print(f"{n}/{len(entrades)} entrades, {len(locutors)} clips", flush=True)

    for veu, linies in metadata.items():
        (args.eixida / veu / "metadata.csv").write_text("\n".join(linies) + "\n", encoding="utf-8")
    (args.eixida / "locutors.tsv").write_text("id\tveu\tf0\tsegons\n" + "\n".join(locutors) + "\n",
                                              encoding="utf-8")
    (args.eixida / "reservades.tsv").write_text("paraula\tfonemes\tfitxer\tveu\n" + "\n".join(reserva) + "\n",
                                                encoding="utf-8")
    (args.eixida / "descartades.tsv").write_text("fitxer\tfonetica\tmotiu\n" + "\n".join(descartades) + "\n",
                                                 encoding="utf-8")
    segons = sum(float(l.split("\t")[3]) for l in locutors)
    per_veu = ", ".join(f"{veu} {len(l)}" for veu, l in metadata.items())
    print(f"{len(locutors)} clips ({segons / 3600:.1f} h; {per_veu}), {len(reserva)} reservades, "
          f"{len(descartades)} descartades → {args.eixida}")


if __name__ == "__main__":
    main()
