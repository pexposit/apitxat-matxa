# Reducción del *apitxat* en una voz sintética valenciana (Matxa / Matcha-TTS)

Código, datos de evaluación y fuentes del artículo *«Reducción del apitxat en una voz sintética valenciana mediante ajuste fino multilocutor de Matcha-TTS con embeddings de locutor mezclados y congelados»*.

> **English summary.** The Valencian voice *Gina* of the Matxa TTS model (Matcha-TTS, BSC) devoices the voiced sibilants /z/, /dz/, /ʒ/, /dʒ/ (*apitxat*), which the Valencian standard keeps voiced. We measure voicing with a low-frequency energy-drop metric and fine-tune Matxa on ~90k recordings from the *Diccionari normatiu valencià* plus Gina clips. Gina's voiced/voiceless contrast goes from ~3 dB to 16.5 dB, which is within the range of the human speakers, while speaker similarity (0.93), estimated quality (PESQ ≈ 4.06) and WER (5.7 %) are preserved. The fine-tune uses three steps:
>
> 1. remove Gina's clips that contain voiced sibilants;
> 2. initialise her speaker embedding as a 50/50 mix with a standard-accent speaker;
> 3. **freeze** that embedding during training.

## Contenido

```
articulo/                 artículo en LaTeX (IEEEtran), bibliografía y figuras
  figuras/genera_figuras.py   regenera las tres figuras a partir de los resultados
scripts/
  setup-matxa.sh          prepara el entorno: .venv-voz, espeak-ng de projecte-aina, Matcha-TTS dev-cat,
                          Matxa multiaccent y vocoder alVoCat
  descarrega-dnv.py       descarga del DNV (transcripción AVL y audio de cada entrada) — ver el aviso de «Datos»
  matxa_entorn.py         carga de modelo/vocoder, fonemización (espeak o AVL) y síntesis
  lexic_avl.py            léxico AVL (DNV → símbolos de Matxa) y fonemizador «avl»
  prepara-dnv.py          grabaciones del DNV → clips por locutor (separados por F0) con fonemas AVL
  prepara-matxa.py        filelists de Matxa (ruta|locutor|fonemas)
  finetune-matxa.py       ajuste fino; --barreja A=B:ALFA (mezcla del embedding) y --congela-spk (congelación)
  entrena-avl.sh          receta completa: SENSE_SONORES=1 (v2), BARREJA=7=5:0.5 (v3), CONGELA=7 (v4)
  sintetitza-matxa.py     síntesis de prueba con un punto de control
  sintetitza-epoques.py   sintetiza unas frases con todos los puntos de control de un entrenamiento
  mescla-locutors.py      interpolación del embedding en inferencia (Fig. 1)
  hornea-barreja.py       guarda un punto de control con el embedding mezclado (y sin el optimizador)
  analitza-apitxat.py     métrica de caída en baja frecuencia y contraste sonora/sorda
  frases-sonores.json     las cinco oraciones de prueba (26 sibilantes)
  frases-avl.json         oraciones de control de la pronunciación AVL
  avaluacio/timbre.py     similitud de locutor (ECAPA-TDNN, SpeechBrain)
  avaluacio/qualitat.py   PESQ/STOI estimados (TorchAudio-Squim) y WER
  avaluacio/prosodia.py   variación de F0 en semitonos (pYIN)
resultados/               medidas del artículo (las tablas y figuras salen de aquí)
```

Los comentarios y los nombres del código están en valenciano, el idioma del proyecto del que procede (ParlaVal).

## Reproducir

```bash
bash scripts/setup-matxa.sh                     # entorno y modelos base (GPU NVIDIA)

# Datos (ver «Datos» más abajo; el DNV requiere permiso previo de la AVL)
.venv-voz/bin/python scripts/descarrega-dnv.py   # voz-dataset/dnv/: entrades.jsonl y flac/
.venv-voz/bin/python scripts/lexic_avl.py       # léxico AVL
.venv-voz/bin/python scripts/prepara-dnv.py     # clips del DNV

# v4 (la receta del artículo)
NOM=avl-v4 DADES=voz-dataset/matxa-avl-sense-sonores SENSE_SONORES=1 \
  BARREJA=7=5:0.5 CONGELA=7 bash scripts/entrena-avl.sh

# muestras de las oraciones de prueba en cada punto de control
.venv-voz/bin/python scripts/sintetitza-epoques.py --run avl-v4 --spk 7 --spk 5 \
  --frases scripts/frases-sonores.json --carpeta mostres-sonores

# medidas
M=tts/runs/avl-v4/mostres-sonores/epoch=0025
.venv-voz/bin/python scripts/analitza-apitxat.py e25=$M:7
.venv-voz/bin/python scripts/avaluacio/timbre.py --ref <gina-original>:7 e25=$M:7
.venv-voz/bin/python scripts/avaluacio/qualitat.py e25=$M:7
.venv-voz/bin/python scripts/avaluacio/prosodia.py e25=$M:7
```

Las carpetas de audio contienen `frase_NN_spkS.wav`. En inferencia se usan 32 pasos de la ODE, temperatura 0,667 y el fonemizador AVL. Como el decodificador parte de ruido, las medidas varían con la semilla (unos ±1,5 dB en el contraste). En el artículo, cada punto de control se evalúa con tres semillas (`mescla-locutors.py --alfa 0 --llavor N`).

**Servicios externos.**

- `analitza-apitxat.py` necesita un alineador fonético con `align=true`. Por defecto usa la API del DeepLab de la UJI; se cambia con la variable `APITXAT_ALINEADOR`.
- `qualitat.py` necesita un reconocedor compatible con la API de OpenAI. Se configura con las variables `STT_URL` y `STT_MODEL`.

## Datos

- **Diccionari normatiu valencià (AVL).** Se usan las entradas (forma y transcripción AVL) y las grabaciones de pronunciación: 89 796 clips, 25,8 h, con una voz femenina y una masculina. `descarrega-dnv.py` las obtiene consultando la web del DNV, y `prepara-dnv.py` y `lexic_avl.py` las procesan. Estos datos **no se redistribuyen** aquí; pertenecen a la Acadèmia Valenciana de la Llengua.

> ⚠️ **Aviso.** El DNV y sus grabaciones son propiedad de la Acadèmia Valenciana de la Llengua. **Antes de usar `descarrega-dnv.py` o cualquier dato del DNV hay que pedir permiso a la AVL** y respetar sus condiciones de uso. El script se publica únicamente para documentar el procedimiento del artículo. Los autores de este repositorio **no se hacen responsables del uso** que terceros hagan del script o de los datos del DNV. Identifícate en las peticiones con la variable `DNV_AGENT`.
- **Clips de Gina.** Son 2 038 clips sintéticos de palabras cortas generados con Matxa (locutora 7) y etiquetados con eSpeak. No se incluyen.
- **`resultados/`.**
  - `contraste-por-epoca.tsv`: contraste, similitud, PESQ, STOI y WER de cada punto de control con la semilla fija.
  - `candidatos-3-semillas.txt`: los candidatos con tres semillas.
  - `sistemas-base.json`: los sistemas de partida y las grabaciones del DNV.
  - `barrido-mezcla.json`: la interpolación de la Fig. 1.
  - `per-checkpoint/`: el detalle por palabra.

## Licencias

- El código de entrenamiento se apoya en **Matcha-TTS** (rama `dev-cat` de BSC) y en el modelo **matxa-tts-cat-multiaccent** (GPL-3.0).
- Las voces de **LaFrescat**, entre ellas Gina, son **solo para uso no comercial y de investigación**. Los modelos ajustados heredan esta restricción y no se distribuyen aquí.
- Este repositorio se publica bajo la licencia **GPL-3.0** (ver `LICENSE`), la misma que Matxa. No cubre los datos del DNV ni las voces de LaFrescat.

## Cita

```bibtex
@inproceedings{apitxat-matxa,
  title  = {Reducción del apitxat en una voz sintética valenciana mediante ajuste fino multilocutor
            de Matcha-TTS con embeddings de locutor mezclados y congelados},
  author = {[Autores]},
  year   = {2026}
}
```
