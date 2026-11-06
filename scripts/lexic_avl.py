#!/usr/bin/env python3
"""Lèxic de pronúncia de l'AVL (DNV) per a Matxa, en lloc d'espeak on discrepen.

`descarrega-dnv.py` guarda la transcripció normativa de cada entrada
(`[kolɛ́ʤi] o [kolɛ́ʤe]`). Ací es passa als símbols que Matxa va aprendre
d'espeak `ca-va`: africades en dos símbols (ʧ→tʃ, ʤ→dʒ, ʣ→dz, ʦ→ts) i
l'accent agut com a `ˈ` just davant de la vocal tònica, on el posa espeak
(`tʃikˈet`), així que no cal sil·labificar.

Ús com a mòdul: `matxa_entorn.fonemitza` crida `aplica(frase, fonemes_espeak)`
quan MATXA_FONEMITZADOR=avl. Les paraules on l'AVL i espeak coincidixen es
queden com les ha dit espeak dins de la frase; en les altres es posa la forma
AVL amb el context d'espeak (àtona, apòstrof, lenició inicial, enllaç final).

Com a script, construïx voz-dataset/dnv/lexic.tsv i fa l'informe de
discrepàncies amb espeak:

  .venv-voz/bin/python scripts/lexic_avl.py
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
DNV = LAB / "voz-dataset" / "dnv"
LEXIC = Path(os.environ.get("MATXA_LEXIC", DNV / "lexic.tsv"))

AFRICADES = {"ʧ": "tʃ", "ʤ": "dʒ", "ʣ": "dz", "ʦ": "ts", "µ": "ɱ"}  # µ: errata del DNV en «triumfant»
VOCALS = set("aeiouɛɔə")
ACCENT = "́"
LENICIO = {"b": "β", "d": "ð", "\u0261": "ɣ"}
RE_PARAULA = re.compile(r"[a-zàèéíïòóúüç·'’-]+")


def a_matxa(ipa: str) -> str:
    """`kolɛ́ʤi` → `kolˈɛdʒi`; `ájɣwa` → `ˈajɣwa`."""
    eixida = []
    for c in unicodedata.normalize("NFD", ipa.strip()):
        if c == ACCENT:
            if eixida and eixida[-1] in VOCALS:
                eixida.insert(len(eixida) - 1, "ˈ")
            continue
        if unicodedata.category(c) == "Mn" or c in "()‿.":
            continue
        eixida.append(AFRICADES.get(c, "\u0261" if c == "g" else c))  # g → ɡ IPA, com espeak
    return "".join(eixida)


def femeni(lema: str, morfema: str) -> str:
    """Grafia del femení: «metge» + «-essa» → «metgessa»; «xiquet» + «-a» → «xiqueta»."""
    sufix = morfema.strip().lstrip("-")
    if not sufix or " " in sufix:
        return ""
    i = lema.rfind(sufix[0])
    return lema[:i] + sufix if i > 0 and len(lema) - i <= 3 else lema + sufix


def plural(paraula: str) -> str:
    """Grafia del plural regular: «diner» → «diners», «vaca» → «vaques»."""
    for singular, pl in (("ca", "ques"), ("ga", "gues"), ("ça", "ces"), ("ja", "ges"), ("gua", "gües"),
                         ("qua", "qües"), ("a", "es")):
        if paraula.endswith(singular):
            return paraula[: -len(singular)] + pl
    return paraula + "s"


def formes_amb_paraula(e: dict) -> list[tuple[str, str, str]]:
    """(paraula, IPA de l'AVL, origen) de cada transcripció de l'entrada, en l'ordre en què es diuen.

    «o» és una variant de la forma d'abans: [dinéɾ] (pl. [dinéɾs] o [dinés]) → diner,
    diners, diners. Les etiquetes que no se sap a quina paraula van tornen paraula buida.
    """
    lema = e.get("lema", "").lower()
    eixida, actual = [], ("", "")
    for f in e.get("formes", []):
        etiqueta = f["etiqueta"]
        if etiqueta == "":
            actual = (lema, "lema")
        elif etiqueta == "f.":
            actual = (femeni(lema, e.get("morfema", "")), "femeni")
        elif etiqueta == "pl.":
            actual = (plural(lema), "plural")
        elif etiqueta != "o":
            actual = (etiqueta.lower(), "forma") if RE_PARAULA.fullmatch(etiqueta) else ("", "")
        eixida.append((actual[0], f["ipa"], actual[1]))
    return eixida


def entrades_lexic(entrades: list[dict]) -> list[tuple[str, list[str], str, dict]]:
    """(paraula, variants en símbols de Matxa, origen, entrada del DNV) de cada forma amb transcripció."""
    eixida = []
    for e in entrades:
        lema = e.get("lema", "")
        if not e.get("formes") or not lema or lema.startswith("-") or lema.endswith("-") or "/" in e["fonetica"]:
            continue
        for paraula, ipa, origen in formes_amb_paraula(e):
            if paraula:
                eixida.append((paraula, [a_matxa(ipa)], origen, e))
        # Femení regular en -a quan el lema acaba en consonant que no canvia: xiquet → xiqueta.
        if e.get("morfema") == "-a" and lema[-1:] in "tlnr" and not any(f["etiqueta"] == "f." for f in e["formes"]):
            eixida.append((lema.lower() + "a", [a_matxa(e["formes"][0]["ipa"]) + "a"], "femeni", e))
    return eixida


def carrega_lexic(ruta: Path = LEXIC) -> dict[str, list[str]]:
    lexic: dict[str, list[str]] = {}
    if ruta.exists():
        for linia in ruta.read_text(encoding="utf-8").splitlines():
            paraula, variants, *_ = linia.split("\t")
            lexic.setdefault(paraula, variants.split("|"))
    return lexic


@lru_cache(maxsize=1)
def _lexic_de(_data: float) -> dict[str, list[str]]:
    return carrega_lexic()


def _lexic() -> dict[str, list[str]]:
    """El lèxic, tornat a llegir si lexic.tsv canvia (creix mentre baixa el DNV)."""
    return _lexic_de(LEXIC.stat().st_mtime if LEXIC.exists() else 0.0)


def regles(espeak: str, avl: str) -> list[tuple[str, str]]:
    """Canvis d'espeak a l'AVL amb dos sons de context: saβˈeɾ → sabˈeɾ dona («saβ», «sab»)."""
    eixida = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, espeak, avl, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        vell, nou = espeak[i1:i2], avl[j1:j2]
        if i1 >= 2:
            eixida.append((espeak[i1 - 2:i1] + vell, espeak[i1 - 2:i1] + nou))
        elif len(espeak) >= i2 + 2:
            eixida.append((vell + espeak[i2:i2 + 2], nou + espeak[i2:i2 + 2]))
    return eixida


def carrega_correccions(ruta: Path = LEXIC) -> dict[str, list[tuple[str, list[tuple[str, str]]]]]:
    """Per a cada arrel de tres lletres, les paraules on espeak s'equivoca i com es corregix."""
    index: dict[str, list[tuple[str, list[tuple[str, str]]]]] = collections.defaultdict(list)
    if ruta.exists():
        for linia in ruta.read_text(encoding="utf-8").splitlines():
            paraula, variants, _, espeak, *_ = linia.split("\t") + [""]
            avl = variants.split("|")
            if espeak and not any(sense_accent(v) == sense_accent(espeak) for v in avl) and len(paraula) >= 3:
                index[paraula[:3]].append((paraula, regles(espeak, avl[0])))
    return index


@lru_cache(maxsize=1)
def _correccions_de(_data: float) -> dict[str, list[tuple[str, list[tuple[str, str]]]]]:
    return carrega_correccions()


def deriva(paraula: str, aillada: str, index: dict | None = None) -> str:
    """Forma no recollida al DNV (trobem, estaven, sabia): els canvis d'espeak a l'AVL de les
    paraules de la mateixa arrel (trobar tɾoβˈaɾ → tɾobˈaɾ) on el context coincidix."""
    if index is None:
        index = _correccions_de(LEXIC.stat().st_mtime if LEXIC.exists() else 0.0)
    eixida = aillada
    for altra, canvis in index.get(paraula[:3], []):
        comu = next((i for i, (x, y) in enumerate(zip(paraula, altra)) if x != y), min(len(paraula), len(altra)))
        if comu < min(4, len(altra) - 2) or comu < len(altra) - 3:
            continue
        for vell, nou in canvis:
            # Obertura de vocals (sort → sord?): només entre paraules que compartixen quatre lletres.
            obertura = any(x != y and {x, y} <= set("eɛoɔ") for x, y in zip(vell, nou))
            if obertura and comu < 4:
                continue
            if vell in eixida:
                eixida = eixida.replace(vell, nou, 1)
    return eixida if eixida != aillada else ""


def busca(paraula: str, lexic: dict[str, list[str]]) -> list[str]:
    """Variants AVL d'una paraula; plurals regulars a partir del singular."""
    if paraula in lexic:
        return lexic[paraula]
    for grafia, singular in (("ques", "ca"), ("gues", "ga"), ("ces", "ça"), ("ges", "ja"),
                             ("gües", "gua"), ("qües", "qua"), ("es", "a")):
        if paraula.endswith(grafia) and (arrel := paraula[: -len(grafia)] + singular) in lexic:
            return [v[:-1] + "es" for v in lexic[arrel] if v.endswith("a")]
    if paraula.endswith("s") and paraula[:-1] in lexic:
        return [v + "s" for v in lexic[paraula[:-1]] if v[-1:] in "tlnrkpmeiou" and not v.endswith(("ˈe", "ˈo"))]
    return []


def sense_accent(s: str) -> str:
    return s.replace("ˈ", "").replace("ˌ", "")


def amb_context(avl: str, aillada: str, frase: str) -> str:
    """Posa a la forma AVL el que espeak canvia dins de la frase respecte de la paraula aïllada."""
    prefix = ""
    nucli = sense_accent(aillada)
    pla = sense_accent(frase)
    if nucli and pla.endswith(nucli[1:]) and len(pla) > len(nucli):  # apòstrof: l'home, d'anglés
        prefix = pla[: len(pla) - len(nucli)]
        if pla[len(prefix)] != nucli[0] and nucli[0] in LENICIO:
            avl = avl.replace(nucli[0], LENICIO[nucli[0]], 1) if avl.lstrip("ˈ").startswith(nucli[0]) else avl
    elif nucli and pla and pla[0] != nucli[0] and LENICIO.get(nucli[0]) == pla[0] and avl.lstrip("ˈ").startswith(nucli[0]):
        avl = avl.replace(nucli[0], pla[0], 1)
    if "ˈ" not in frase:
        avl = sense_accent(avl)
    # Enllaç final (s→z davant de vocal, i→j…): les últimes lletres que espeak canvia.
    comu = 0
    while comu < min(len(pla), len(nucli)) and pla[-1 - comu] == nucli[-1 - comu]:
        comu += 1
    if comu == 0:
        for n in (1, 2):
            if len(pla) > n and len(nucli) > n and pla[:-n].endswith(nucli[-n - 2:-n]) and avl.endswith(nucli[-n:]):
                avl = avl[:-n] + pla[-n:]
                break
    return prefix + avl


@lru_cache(maxsize=50000)
def espeak_aillada(paraula: str) -> str:
    from matcha.text.cleaners import catalan_valencia_cleaners

    return catalan_valencia_cleaners(paraula).strip(" .,;:!?")


def aplica(text: str, fonemes: str, lexic: dict[str, list[str]] | None = None) -> str:
    """Canvia en la fonemització d'espeak de `text` les paraules on l'AVL diu una altra cosa."""
    lexic = _lexic() if lexic is None else lexic
    if not lexic:
        return fonemes
    paraules = text.lower().split()
    trossos = fonemes.split(" ")
    if len(paraules) != len(trossos):  # xifres, abreviatures…: espeak tal qual
        return fonemes
    eixida = []
    for paraula, tros in zip(paraules, trossos):
        m = re.match(r"^([^\wàèéíïòóúüç·]*)(.*?)([^\wàèéíïòóúüç·]*)$", tros)
        _, fon, puntuacio = m.groups() if m else ("", tros, "")
        neta = paraula.strip(".,;:!?¡¿«»\"()…—").replace("’", "'")
        cap, *enclitics = neta.split("'")[-1].split("-")
        variants = busca(cap, lexic)
        if not variants and fon and len(cap) >= 3:
            derivada = deriva(cap, espeak_aillada(cap))
            variants = [derivada] if derivada else []
        if not variants or not fon:
            eixida.append(tros)
            continue
        aillada = espeak_aillada(cap)
        if any(sense_accent(v) == sense_accent(aillada) for v in variants):
            eixida.append(tros)
            continue
        cua = ""
        if enclitics:  # esperar-te: només es canvia el verb, el pronom es queda com el diu espeak
            sense = sense_accent(aillada)
            i = fon.replace("ˌ", "").find(aillada) if aillada in fon else -1
            if i < 0 or not sense:
                eixida.append(tros)
                continue
            cua, fon = fon[i + len(aillada):], fon[: i + len(aillada)]
        eixida.append(amb_context(variants[0], aillada, fon) + cua + puntuacio)
    return " ".join(eixida)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrades", type=Path, default=DNV / "entrades.jsonl")
    ap.add_argument("--eixida", type=Path, default=LEXIC)
    ap.add_argument("--informe", type=Path, default=DNV / "discrepancies-espeak.tsv")
    ap.add_argument("--comparacio", type=Path, default=DNV / "comparacio.json",
                    help="discrepàncies amb l'àudio del DNV, per a public/fonetica-avl.html")
    args = ap.parse_args()

    import matxa_entorn  # noqa: F401  (entorn d'espeak per a catalan_valencia_cleaners)

    entrades = [json.loads(l) for l in args.entrades.read_text(encoding="utf-8").splitlines() if l.strip()]
    files = entrades_lexic(entrades)
    # Una línia per paraula amb totes les variants: «dona» és [dɔ́na] (nom) i [dóna] (de donar).
    # Les de [h] són interjeccions («ha!», «he!») que no han de tapar el verb.
    per_paraula: dict[str, tuple[list[str], str, dict]] = {}
    for paraula, variants, origen, entrada in files:
        actual, _, _ = per_paraula.setdefault(paraula, ([], origen, entrada))
        actual += [v for v in variants if v not in actual and "h" not in v]
    linies, discrepancies, comparacio = [], [], []
    tipus = collections.Counter()
    for paraula, (variants, origen, entrada) in per_paraula.items():
        if not variants:
            continue
        es = espeak_aillada(paraula)
        igual = any(sense_accent(v) == sense_accent(es) for v in variants)
        linies.append(f"{paraula}\t{'|'.join(variants)}\t{origen}\t{es}")
        if not igual:
            discrepancies.append(f"{paraula}\t{variants[0]}\t{es}")
            comparacio.append({"paraula": paraula, "avl": variants, "espeak": es, "origen": origen,
                               "lema": entrada["lema"], "fonetica": entrada["fonetica"],
                               "audio": entrada.get("fitxer", "")})
            a, b = sense_accent(variants[0]), sense_accent(es)
            if len(a) == len(b):
                for x, y in zip(a, b):
                    if x != y:
                        tipus[f"AVL {x} / espeak {y}"] += 1
            else:
                tipus["llargària diferent"] += 1
    args.eixida.write_text("\n".join(linies) + "\n", encoding="utf-8")
    args.informe.write_text("paraula\tavl\tespeak\n" + "\n".join(discrepancies) + "\n", encoding="utf-8")
    args.comparacio.write_text(json.dumps(comparacio, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(linies)} paraules en {args.eixida}; espeak discrepa en {len(discrepancies)} "
          f"({len(discrepancies) / max(len(linies), 1):.0%})")
    for t, n in tipus.most_common(15):
        print(f"  {n:6d}  {t}")


if __name__ == "__main__":
    main()
