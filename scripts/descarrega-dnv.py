#!/usr/bin/env python3
"""Descarrega del DNV de l'AVL la transcripció fonètica i l'àudio de cada entrada.

AVÍS: el Diccionari normatiu valencià i les seues gravacions són de l'Acadèmia Valenciana de la
Llengua. Abans d'executar este script cal demanar permís a l'AVL i respectar les seues condicions
d'ús. Els autors d'este repositori no es fan responsables de l'ús que se'n faça.

Cada pàgina `lexicval/?paraula=<p>` porta el lema, la transcripció fonètica
normativa (`[kolɛ́ʤi] o [kolɛ́ʤe]`, `[méʤe] (f. [meʤésa])`, `[menʤáɾ] (menja
[ménʤa])`) i un mp3 d'estudi. El DNV no té índex consultable: es pregunta per
cada lema de la llista de freqüències de Softcatalà (els més usats primer). Si
la paraula no hi és, el DNV mostra l'entrada més pròxima; es guarda igual si és
nova. Reprén on ho va deixar: llig `entrades.jsonl` i salta el que ja s'ha
consultat.

  .venv-voz/bin/python scripts/descarrega-dnv.py --limit 200
  setsid nohup .venv-voz/bin/python scripts/descarrega-dnv.py --prioritat voz-dataset/dnv/paraules-app.txt \\
      > voz-dataset/dnv/descarrega.log 2>&1 &

Eixida (voz-dataset/dnv/): entrades.jsonl (una línia per entrada o consulta
sense resultat), consultes.txt (paraules ja preguntades), veines.txt (entrades
que el DNV llista com a resultats i encara no s'han preguntat) i flac/<nom>.flac.
Els mp3 (estèreo, 48 kHz, 320 kbps) es guarden en FLAC mono a 22 050 Hz, el
format de Matxa: un terç de lloc. Es para si el disc es queda amb menys de
`--minim-gb` lliures.

Ordre de les consultes: `--prioritat` (p. ex. les paraules de l'app; el DNV
lematitza: «cases» → casa), els lemes per freqüència, la resta de lemes dels
diccionaris font de Softcatalà i, en bola de neu, les veïnes que van apareixent,
fins que no n'ix cap de nova.
"""
from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import html
import json
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.parse
from collections import deque
from pathlib import Path

import requests

LAB = Path(__file__).resolve().parent.parent
URL_PAGINA = "https://www.avl.gva.es/lexicval/?paraula={}"
URL_AUDIO = "https://www.avl.gva.es/lexicval/javax.faces.resource/audi.xhtml?ln=audiResources&nom={}&appName=dnv"
URL_SC = "https://raw.githubusercontent.com/Softcatala/catalan-dict-tools/master/"
LLISTES_SC = ["frequencies/frequencies-dict-lemmas.txt", "diccionari-arrel/dnv-fdic.txt",
              "diccionari-arrel/verbs-fdic.txt", "diccionari-arrel/adjectius-fdic.txt",
              "diccionari-arrel/noms-fdic.txt", "diccionari-arrel/adverbis-lt.txt"]
AGENT = os.environ.get("DNV_AGENT", "apitxat-matxa (recerca)")  # identifica-t'hi

RE_LEMA = re.compile(r"<div class='lema'>(.*?)</div>", re.S)
RE_IPA = re.compile(r"\[([^\]]+)\]")
RE_RESULTAT = re.compile(r'<tr data-ri="\d+"[^>]*><td role="gridcell"[^>]*>(.*?)</td>', re.S)


_local = threading.local()


def obte(url: str, intents: int = 4) -> bytes | None:
    """GET amb una connexió reaprofitada per fil (keep-alive): sense un TLS nou a cada petició."""
    if not hasattr(_local, "sessio"):
        _local.sessio = requests.Session()
        _local.sessio.headers["User-Agent"] = AGENT
    for i in range(intents):
        try:
            r = _local.sessio.get(url, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            pass
        time.sleep(5 * 2 ** i)
    # Sense resposta (xarxa o servidor caiguts): millor parar que marcar la paraula com a feta.
    raise ConnectionError(f"sense resposta després de {intents} intents: {url}")


def a_flac(mp3: bytes, desti: Path) -> bool:
    """mp3 → FLAC mono 22 050 Hz, amb ffmpeg o, si no n'hi ha (deeplab), amb soundfile i librosa."""
    temporal = desti.with_name(f".{threading.get_ident()}.{desti.name}")
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "22050",
                            "-sample_fmt", "s16", "-f", "flac", str(temporal)], input=mp3, capture_output=True)
        ok = r.returncode == 0
    else:
        import io

        import librosa
        import soundfile as sf

        try:
            y, sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
            sf.write(temporal, librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=22050), 22050,
                     format="FLAC", subtype="PCM_16")
            ok = True
        except (RuntimeError, ValueError, sf.LibsndfileError):
            ok = False
    if ok:
        os.replace(temporal, desti)
    return ok


def text_pla(fragment: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(html.unescape(re.sub(r"<[^>]+>", "", fragment))))


def formes(fonetica: str) -> list[dict]:
    """`[méʤe] (f. [meʤésa])` → [{etiqueta: '', ipa: 'méʤe'}, {etiqueta: 'f.', ipa: 'meʤésa'}]."""
    eixida, abans = [], 0
    for m in RE_IPA.finditer(fonetica):
        etiqueta = fonetica[abans:m.start()].strip(" (),;")
        eixida.append({"etiqueta": etiqueta, "ipa": m.group(1).strip()})
        abans = m.end()
    return eixida


def analitza(pagina: str) -> list[dict]:
    entrades = []
    for bloc in RE_LEMA.findall(pagina):
        lema = re.search(r"<span class='lema'>(.*?)</span>", bloc)
        if not lema:
            continue
        sup = re.search(r"<span class='superindex'>(.*?)</span>", bloc)
        morfema = re.search(r"<span class='morfema negrita'>(.*?)</span>", bloc)
        inici = bloc.find("<span class='trfonetica'>")
        final = min((i for i in (bloc.find("<audio", inici), bloc.find("<button", inici)) if i >= 0),
                    default=len(bloc))
        fonetica = text_pla(bloc[inici:final]) if inici >= 0 else ""
        fonetica = fonetica.replace("\xa0", " ")
        mp3 = re.search(r"nom=([^&\"]+\.mp3)", bloc)
        entrades.append({
            "lema": text_pla(lema.group(1)).strip(),
            "superindex": text_pla(sup.group(1)).strip() if sup else "",
            "morfema": text_pla(morfema.group(1)).strip() if morfema else "",
            "fonetica": fonetica.strip(),
            "formes": formes(fonetica),
            "mp3": mp3.group(1) if mp3 else "",
        })
    return entrades


def veines(pagina: str) -> list[str]:
    """Entrades de la llista «Resultats» («xic 1» → «xic»)."""
    return [re.sub(r"\s+\d+$", "", text_pla(x).strip()) for x in RE_RESULTAT.findall(pagina)]


def lemes_softcatala(fitxer: Path) -> list[str]:
    """Primera paraula de cada línia: «lema, freq», «lema=categories…», «agut aguda=…»."""
    eixida = []
    for linia in fitxer.read_text(encoding="utf-8").splitlines():
        lema = re.split(r"[=, ]", linia.lstrip("#"), maxsplit=1)[0].strip()
        if lema:
            eixida.append(lema)
    return eixida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eixida", type=Path, default=LAB / "voz-dataset" / "dnv")
    ap.add_argument("--prioritat", type=Path, action="append", default=[],
                    help="fitxer de paraules (una per línia) que es pregunten primer")
    ap.add_argument("--limit", type=int, default=0, help="nombre màxim de consultes noves (0 = totes)")
    ap.add_argument("--pausa", type=float, default=0.25, help="segons entre peticions de cada fil")
    ap.add_argument("--fils", type=int, default=1,
                    help="consultes alhora (el servidor del DNV les atén d'una en una: més fils no van més ràpid)")
    ap.add_argument("--sense-audio", action="store_true")
    ap.add_argument("--minim-gb", type=float, default=3.0, help="espai lliure mínim al disc")
    args = ap.parse_args()

    (args.eixida / "flac").mkdir(parents=True, exist_ok=True)
    (args.eixida / "llistes").mkdir(exist_ok=True)
    candidates = []
    for fitxer in args.prioritat:
        candidates += [l.strip() for l in fitxer.read_text(encoding="utf-8").splitlines()]
    for ruta in LLISTES_SC:
        local = args.eixida / "llistes" / Path(ruta).name
        if not local.exists():
            local.write_bytes(obte(URL_SC + ruta) or b"")
        candidates += lemes_softcatala(local)

    consultes_f = args.eixida / "consultes.txt"
    veines_f = args.eixida / "veines.txt"
    fetes = set(consultes_f.read_text(encoding="utf-8").splitlines()) if consultes_f.exists() else set()
    if veines_f.exists():
        candidates += veines_f.read_text(encoding="utf-8").splitlines()
    vistes = set()
    entrades_f = args.eixida / "entrades.jsonl"
    if entrades_f.exists():
        for linia in entrades_f.read_text(encoding="utf-8").splitlines():
            e = json.loads(linia)
            vistes.add((e.get("lema"), e.get("superindex")))

    cua = deque(dict.fromkeys(c for c in candidates if c and " " not in c and c not in fetes))
    en_cua = set(cua)
    print(f"{len(cua)} paraules per preguntar, {len(vistes)} entrades ja baixades", flush=True)
    def consulta(lema: str) -> tuple[str, str, list[dict]]:
        pagina = obte(URL_PAGINA.format(urllib.parse.quote(lema)))
        time.sleep(args.pausa)
        text = pagina.decode("utf-8", "replace") if pagina else ""
        entrades = analitza(text)
        for e in entrades:
            if e["mp3"] and not args.sense_audio:
                desti = args.eixida / "flac" / Path(urllib.parse.unquote(e["mp3"])).with_suffix(".flac").name
                if not desti.exists():
                    audio = obte(URL_AUDIO.format(e["mp3"]))
                    time.sleep(args.pausa)
                    if audio:
                        a_flac(audio, desti)
                e["fitxer"] = desti.name if desti.exists() else ""
        return lema, text, entrades

    noves = 0
    with entrades_f.open("a", encoding="utf-8") as ent, consultes_f.open("a", encoding="utf-8") as con, \
            veines_f.open("a", encoding="utf-8") as vei, ThreadPoolExecutor(args.fils) as fils:
        en_curs = set()
        while cua or en_curs:
            ple = shutil.disk_usage(args.eixida).free < args.minim_gb * 1e9
            if ple and not en_curs:
                print(f"parat: menys de {args.minim_gb} GB lliures al disc", flush=True)
                break
            while cua and len(en_curs) < args.fils and not ple and not (args.limit and noves >= args.limit):
                en_curs.add(fils.submit(consulta, cua.popleft()))
                noves += 1
            if not en_curs:
                break
            fetes_ara, en_curs = wait(en_curs, return_when=FIRST_COMPLETED)
            for futur in fetes_ara:
                lema, text, entrades = futur.result()
                for e in entrades:
                    clau = (e["lema"], e["superindex"])
                    if clau in vistes:
                        continue
                    vistes.add(clau)
                    e["consulta"] = lema
                    ent.write(json.dumps(e, ensure_ascii=False) + "\n")
                ent.flush()
                for v in veines(text):
                    if v and " " not in v and v not in fetes and v not in en_cua:
                        cua.append(v)
                        en_cua.add(v)
                        vei.write(v + "\n")
                con.write(lema + "\n")
                con.flush()
                fetes.add(lema)
                if len(fetes) % 500 == 0:
                    print(f"{len(fetes)} consultes, {len(vistes)} entrades, {len(cua)} en cua", flush=True)
    print(f"fet: {noves} consultes noves, {len(vistes)} entrades")


if __name__ == "__main__":
    main()
