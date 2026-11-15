#!/usr/bin/env bash
# Все эксперименты из отчёта одной командой:
#     bash freq_exp/experiments.sh E05          # один эксперимент
#     bash freq_exp/experiments.sh E01 E02 E03  # несколько подряд
#     bash freq_exp/experiments.sh list         # показать список
# Запускать из папки PiSA-SR-main. Пути задаются переменными окружения (или правкой значений ниже).
set -euo pipefail

SD=${SD:-/kaggle/working/sd_v2_1_full}                 # Stable Diffusion 2.1-base
INIT=${INIT:-preset/models/pisa_sr.pkl}                # релизный чекпоинт PiSA-SR
RAM=${RAM:-}                                           # ram_swin_large_14m.pth (пусто = без RAM)
TRAIN_LQ=${TRAIN_LQ:-}
TRAIN_HQ=${TRAIN_HQ:-}
VAL_LQ=${VAL_LQ:-}
VAL_HQ=${VAL_HQ:-}
OUT=${OUT:-experiments/freq}                           # корень для результатов
STEPS=${STEPS:-4000}                                   # шагов на этап
EVAL_EVERY=${EVAL_EVERY:-500}                          # валидация каждые столько шагов
ACCUM=${ACCUM:-1}                                      # накопление градиента
EMA=${EMA:-0}                                          # EMA весов LoRA (например 0.999)
FDL_CROP=${FDL_CROP:-256}                              # окно для FDL (0 = весь кроп 512, нужно больше памяти)
AUTO=${AUTO:-0.3}                                      # калибровка весов частотных членов (0 = выключить)
VAE_DEC=${VAE_DEC:-}                                   # К6: дообученный декодер VAE (пусто = исходный)
VAE_STEPS=${VAE_STEPS:-4000}                           # К6: шагов дообучения декодера

# стартовые веса (по нормам градиента на бикубическом тесте; при AUTO>0 уточняются автоматически)
W_FFL=1.0; W_FSL=6e-9; W_FDL=0.003; W_EDGE=0.0005; W_BANDS=0.0025

COMMON=(--pretrained_model_path "$SD" --lq_dir "$TRAIN_LQ" --hq_dir "$TRAIN_HQ"
        --val_lq_dir "$VAL_LQ" --val_hq_dir "$VAL_HQ" --pix_steps "$STEPS" --sem_steps "$STEPS" --eval_every "$EVAL_EVERY"
        --grad_accum "$ACCUM" --ema_decay "$EMA" --fdl_crop "$FDL_CROP" --auto_weight_ratio "$AUTO")
[ -n "$RAM" ] && COMMON+=(--ram_path "$RAM")
[ -n "$VAE_DEC" ] && COMMON+=(--vae_decoder_ckpt "$VAE_DEC")

SEM_ONLY=(--stage sem --sem_init_ckpt "$INIT")         # sem-этап сразу с релизного чекпоинта
BOTH=(--stage both --init_ckpt "$INIT")                # pix-этап, затем sem-этап
V1=(--pix_ffl $W_FFL --sem_fdl $W_FDL)                 # В1: FFL в pix-этапе, FDL в sem-этапе
DAW=(--pix_daw 1.0 --sem_daw 1.0)                      # В3: карта деталей для L2 и LPIPS

run() { local name=$1; shift
        [ -n "$VAE_DEC" ] && name="${name}_vaedec"     # запуски с новым декодером пишутся в отдельную папку
        : "${TRAIN_LQ:?укажите TRAIN_LQ}" "${TRAIN_HQ:?укажите TRAIN_HQ}" "${VAL_LQ:?укажите VAL_LQ}" "${VAL_HQ:?укажите VAL_HQ}"
        echo "=== $name: $*"
        python freq_exp/train_freq.py "${COMMON[@]}" --output_dir "$OUT/$name" "$@"; }

# К6: дообучение декодера VAE (без UNet и без LQ)
vae() { local name=$1; shift
        : "${TRAIN_HQ:?укажите TRAIN_HQ}" "${VAL_HQ:?укажите VAL_HQ}"
        echo "=== $name: $*"
        python freq_exp/train_vae_decoder.py --pretrained_model_path "$SD" --hq_dir "$TRAIN_HQ" --val_hq_dir "$VAL_HQ" \
            --steps "$VAE_STEPS" --auto_weight_ratio "$AUTO" --output_dir "$OUT/$name" "$@"; }

exp() { case $1 in
  # --- шум базовой линии: исходные потери PiSA-SR, два сида ---
  E00) run E00_base_pixsem         "${BOTH[@]}" --seed 123 ;;   # контроль для E05–E15: оба этапа, исходные потери
  E01) run E01_base_seed123       "${SEM_ONLY[@]}" --seed 123 ;;
  E02) run E02_base_seed456        "${SEM_ONLY[@]}" --seed 456 ;;
  # --- К1: FDL как в статье ---
  E03) run E03_K1_fdl_instead_lpips "${SEM_ONLY[@]}" --sem_fdl 0.01 --sem_lpips 0 --auto_weight_ratio 0 ;;  # FDL — главный член, без калибровки
  E04) run E04_K1_fdl_plus_lpips   "${SEM_ONLY[@]}" --sem_fdl $W_FDL ;;
  # --- В1: позиционная частотная потеря в pix-этапе, FDL в sem-этапе ---
  E05) run E05_V1_ffl_pix_fdl_sem  "${BOTH[@]}" "${V1[@]}" ;;
  E06) run E06_V1_fsl_pix_fdl_sem  "${BOTH[@]}" --pix_fsl $W_FSL --sem_fdl $W_FDL ;;
  E07) run E07_ctrl_ffl_fdl_in_sem "${BOTH[@]}" --sem_ffl $W_FFL --sem_fdl $W_FDL ;;   # контроль: обе в sem-этапе
  # --- К2: В1 + карта деталей ---
  E08) run E08_K2_daw              "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" ;;
  E09) run E09_K2_daw_fdl_regions  "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl ;;
  # --- К3 / К4: надстройки над К2 ---
  E10) run E10_K3_fdl_edges        "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl --sem_fdl_edge $W_EDGE ;;
  E11) run E11_K4_fdl_bands        "${BOTH[@]}" --pix_ffl $W_FFL "${DAW[@]}" --daw_fdl --sem_fdl_bands $W_BANDS ;;
  # --- К5: регуляризатор CSD поверх К2 ---
  E12) run E12_K5_csd_half         "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl --sem_csd 0.5 ;;
  E13) run E13_K5_csd_cfg3         "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl --cfg_csd 3.0 ;;
  E14) run E14_K5_csd_off          "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl --sem_csd 0 ;;
  E15) run E15_K5_csd_daw          "${BOTH[@]}" "${V1[@]}" "${DAW[@]}" --daw_fdl --daw_csd ;;
  # --- К6: декодер VAE. Готовый декодер подключается к любому эксперименту выше: VAE_DEC=<путь> ... E09 ---
  E16) vae E16_K6_vae_l1_lpips ;;                                       # контроль: только L1 + LPIPS
  E17) vae E17_K6_vae_ffl          --w_ffl $W_FFL ;;                    # + Focal Frequency Loss
  E18) vae E18_K6_vae_ffl_fdl      --w_ffl $W_FFL --w_fdl $W_FDL ;;     # + FFL и FDL
  list) grep -E '^[[:space:]]+E[0-9]+\)' "$0" | sed -E 's/^[[:space:]]+(E[0-9]+)\) (run|vae) ([A-Za-z0-9_]+).*/\1  \3/' ;;
  *) echo "неизвестный эксперимент: $1 (см. bash freq_exp/experiments.sh list)"; exit 1 ;;
esac; }

for e in "$@"; do exp "$e"; done
