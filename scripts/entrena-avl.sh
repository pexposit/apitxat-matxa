#!/usr/bin/env bash
# Fine-tune de Matxa `avl-v1`: gina amb la pronúncia de l'AVL (pensat per a executar-se desatès en un servidor amb GPU).
#
# Dades (prepara-dnv.py i clips-gina-curtes-v2):
#   locutor 7 (gina)  clips sintètics de gina, etiquetats amb espeak perquè és el que diuen de veritat;
#                     repetits REPETICIONS vegades perquè no es perden entre les 90 000 del DNV
#   locutor 5 (emma)  la dona del DNV, amb la transcripció AVL
#   locutor 4 (pere)  l'home del DNV, amb la transcripció AVL
# Els sons de l'AVL són els mateixos símbols per a tots: en inferència, gina amb
# MATXA_FONEMITZADOR=avl hauria de dir-los amb el seu timbre.
#
# Si la GPU es compartix amb altres treballs i l'entrenament
# cau (memòria), es reprén de l'últim checkpoint. Abans de cada intent espera que la GPU
# tinga MEMORIA_MIB lliures, per no fer caure els treballs d'altres. Al final deixa tts/runs/$NOM/FET.
#
#   setsid nohup bash scripts/entrena-avl.sh > tts/logs/avl-v1.log 2>&1 < /dev/null &
#
# avl-v2: igual, però traient els clips de gina amb sibil·lants sonores (SENSE_SONORES=1):
# z, dz, ʒ, dʒ. Gina és apitxada: diu [s]/[ts]/[ʃ]/[tʃ] on els fonemes d'espeak posen la sonora,
# i amb eixos clips el model aprén que les sonores de gina són sordes.
#   NOM=avl-v2 DADES=voz-dataset/matxa-avl-sense-sonores SENSE_SONORES=1 \
#     setsid nohup bash scripts/entrena-avl.sh > tts/logs/avl-v2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-voz/bin/python
NOM=${NOM:-avl-v1}
DADES=${DADES:-voz-dataset/matxa-avl}
SENSE_SONORES=${SENSE_SONORES:-0}
REPETICIONS=10
MEMORIA_MIB=${MEMORIA_MIB:-8000}
BARREJA=${BARREJA:-}  # p. ex. 7=5:0.5 (vegeu finetune-matxa.py --barreja)
CONGELA=${CONGELA:-}  # p. ex. 7: l'embedding de gina no s'entrena (finetune-matxa.py --congela-spk)
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MATXA_FONEMITZADOR=avl  # les mostres de cada checkpoint, amb la fonètica AVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -s "$DADES/train.txt" ]; then
  log "filelists"
  # prepara-matxa fonemitza amb MATXA_FONEMITZADOR: per a gina ha de ser espeak.
  MATXA_FONEMITZADOR=espeak $PY scripts/prepara-matxa.py --eixida "$DADES" \
    --font voz-dataset/clips-dnv/dona:5 --font voz-dataset/clips-dnv/home:4 \
    --font voz-dataset/clips-gina-curtes-v2:7 --val 20 || { log "prepara-matxa ha fallat"; exit 1; }
  if [ "$SENSE_SONORES" = 1 ]; then
    # fonemes en la 3a columna: fora qualsevol z o ʒ (també dz, dʒ i les sonoritzades per assimilació)
    for f in train val; do
      awk -F'|' '$2 != 7 || $3 !~ /z|ʒ/' "$DADES/$f.txt" > "$DADES/$f.tmp" && mv "$DADES/$f.tmp" "$DADES/$f.txt"
    done
  fi
  gina=$(grep -c '|7|' "$DADES/train.txt")
  grep '|7|' "$DADES/train.txt" > "$DADES/gina.txt"
  for _ in $(seq 2 "$REPETICIONS"); do cat "$DADES/gina.txt" >> "$DADES/train.txt"; done
  log "train: $(wc -l < "$DADES/train.txt") línies ($gina de gina x$REPETICIONS)"
fi

memoria_lliure() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

for intent in $(seq 1 30); do
  while [ "$(memoria_lliure)" -lt "$MEMORIA_MIB" ]; do
    log "GPU $CUDA_VISIBLE_DEVICES amb $(memoria_lliure) MiB lliures (cal $MEMORIA_MIB): espere 10 minuts"
    sleep 600
  done
  if [ -f "tts/runs/$NOM/checkpoints/last.ckpt" ]; then reprendre=--reprendre; else reprendre=; fi
  log "entrenament, intent $intent $reprendre"
  if $PY scripts/finetune-matxa.py --nom "$NOM" --dades "$DADES" --epoques 30 --cada 2 --batch 16 \
      --spk 7 --spk 5 $reprendre ${BARREJA:+--barreja "$BARREJA"} ${CONGELA:+--congela-spk "$CONGELA"}; then
    date '+%F %T' > "tts/runs/$NOM/FET"
    log "fet"
    exit 0
  fi
  log "ha caigut; torne a provar en 10 minuts"
  sleep 600
done
log "massa intents"; exit 1
