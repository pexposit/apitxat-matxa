"""Entorn comú per a entrenar i sintetitzar amb Matxa (Matcha-TTS de BSC, branca dev-cat).

Tot el que és pesat viu en `speech-lab/tts/` (no es trackeja) i es prepara amb
`scripts/setup-matxa.sh`:

  tts/espeak-ng/                  espeak-ng de projecte-aina compilat (veu `ca-va`)
  tts/Matcha-TTS-dev-cat/         codi d'entrenament i inferència
  tts/models/matxa-tts-cat-multiaccent/checkpoint_epoch=629.ckpt
  tts/models/alvocat-vocos-22khz/ vocoder alVoCat (mel 22,05 kHz → ona)

Cal importar este mòdul ABANS que qualsevol cosa de `matcha`: el mòdul de
cleaners crea els fonemitzadors d'espeak en importar-se i llig l'entorn en
aquell moment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
TTS = LAB / "tts"
ESPEAK = TTS / "espeak-ng"
MATCHA = TTS / "Matcha-TTS-dev-cat"
CKPT_BASE = TTS / "models" / "matxa-tts-cat-multiaccent" / "checkpoint_epoch=629.ckpt"
VOCODER = TTS / "models" / "alvocat-vocos-22khz"

# Locutors del model multiaccent (LaFrescat). El fine-tune reaprofita el de lluc
# (valencià, home) per al nostre narrador: mateix accent i mateix sexe, així
# l'embedding de partida ja està prop de la veu que volem.
SPK_LLUC = 6
SPK_GINA = 7
CLEANER_VAL = "catalan_valencia_cleaners"

os.environ.setdefault("MPLBACKEND", "Agg")  # Matcha dibuixa mels per a tensorboard, sense finestra
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(ESPEAK / "lib" / "libespeak-ng.so"))
os.environ.setdefault("ESPEAK_DATA_PATH", str(ESPEAK / "share" / "espeak-ng-data"))
if str(MATCHA) not in sys.path:
    sys.path.insert(0, str(MATCHA))


def fonemitza(text: str, fonemitzador: str | None = None) -> str:
    """Text valencià → IPA amb espeak `ca-va`, com els filelists de LaFrescat.

    Amb `fonemitzador="avl"` (o MATXA_FONEMITZADOR=avl), les paraules on el DNV
    de l'AVL diu una altra pronúncia (e/ɛ, o/ɔ…) es canvien per la normativa
    (scripts/lexic_avl.py).
    """
    from matcha.text.cleaners import catalan_valencia_cleaners

    fonemes = catalan_valencia_cleaners(text)
    if (fonemitzador or os.environ.get("MATXA_FONEMITZADOR", "espeak")) == "avl":
        import lexic_avl

        return lexic_avl.aplica(text, fonemes)
    return fonemes


def vocoder_del_run(ckpt: Path) -> Path | None:
    """Vocoder afinat per a un fine-tune (tts/runs/<run>/vocoder.bin, scripts/afina-vocoder.py), si n'hi ha."""
    propi = Path(ckpt).resolve().parent.parent / "vocoder.bin"
    return propi if propi.is_file() else None


def carrega_vocoder(device, pesos: Path | None = None):  # noqa: ANN001, ANN201
    """alVoCat, o els pesos d'un vocoder afinat (mateixa arquitectura)."""
    import torch
    from vocos import Vocos

    voc = Vocos.from_hparams(str(VOCODER / "config.yaml"))
    voc.load_state_dict(torch.load(pesos or VOCODER / "pytorch_model.bin", map_location=device))
    return voc.to(device).eval()


def carrega_model(ckpt: Path, device):  # noqa: ANN001, ANN201
    from matcha.models.matcha_tts import MatchaTTS

    return MatchaTTS.load_from_checkpoint(str(ckpt), map_location=device).to(device).eval()


def sintetitza(model, vocoder, text: str, spk: int, device, *, passos: int = 32,  # noqa: ANN001, ANN201
               temperatura: float = 0.667, velocitat: float = 1.0, fonemitzador: str | None = None):
    """Torna l'ona (tensor 1D, 22,05 kHz). `velocitat` < 1 parla més ràpid (length_scale)."""
    import torch
    from matcha.text import text_to_sequence
    from matcha.utils.utils import intersperse

    ids = intersperse(text_to_sequence(fonemitza(text, fonemitzador), ["basic_cleaners"]), 0)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    with torch.inference_mode():
        out = model.synthesise(x, torch.tensor([x.shape[-1]], device=device), n_timesteps=passos,
                               temperature=temperatura, spks=torch.tensor([spk], device=device),
                               length_scale=velocitat)
        return vocoder.decode(out["mel"]).cpu().squeeze()
