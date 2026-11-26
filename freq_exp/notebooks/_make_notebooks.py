"""Генератор ноутбуков для Kaggle. Запуск:  python freq_exp/notebooks/_make_notebooks.py
Ноутбуки создаются рядом с этим файлом. Править удобнее здесь, а не в .ipynb."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update({"outputs": [], "execution_count": None})
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}}}


def md(s): return ("markdown", s)
def code(s): return ("code", s)


INSTRUCTION = """
## Инструкция

1. **Настройки ноутбука (панель справа).** Accelerator — `GPU T4 x2`, Internet — `On`, Persistence — `Files only`.
2. **Подключите данные (Add Input):**
   - обучающие HQ и валидационные HQ (LQ создаются в ноутбуке ячейкой `make_lq`);
   - датасет с Stable Diffusion 2.1-base (папки `unet`, `vae`, `tokenizer`, `text_encoder`) и релизный чекпоинт `pisa_sr.pkl`;
   - код берётся из GitHub (`git clone` в первой ячейке), поэтому Internet — `On`.
3. **Ячейки подготовки.** Первые ячейки клонируют код, собирают папку SD 2.1-base (в датасете нет `scheduler`, конфиг
   создаётся автоматически) и создают LQ. В ячейке «Пути» впишите свои пути; ячейка «Проверка» сразу скажет, чего не хватает.
4. **Запускайте ячейки по порядку.** Обучение печатает строку каждые 20 шагов и валидацию каждые `EVAL_EVERY` шагов (по умолчанию 500).
5. **Время.** Сессия Kaggle ограничена; если этап не успевает, уменьшите `STEPS` или запускайте этапы по отдельности
   (см. раздел «Если сессия оборвалась» в конце).
6. **Сохраните результат.** Последняя ячейка упаковывает логи и чекпоинты в архив. Нажмите `Save Version` →
   `Save & Run All` либо скачайте архив из `/kaggle/working`. Для сравнения экспериментов
   (ноутбук `08_compare_all`) подключите сохранённые результаты как входные данные.

**Что смотреть в результатах**

- `curve.png` и `curve_*.csv` — LPIPS, VoL/GT и ошибка в высокочастотных полосах по шагам;
- `hf_err` — главная метрика: 1 = детали не восстановлены, больше 1 = выдуманы, меньше — лучше;
- `eff_gsd_cm` — эффективное разрешение в сантиметрах (меньше — лучше);
- `train_log_*.csv` — значение, вес и норма градиента каждого члена потерь. Если норма градиента частотного
  члена меньше 5 % от главного (L2 в pix-этапе, LPIPS в sem-этапе), член почти не влияет — увеличьте вес.
"""

BOOT = """# Код из GitHub. Сначала уходим из папки репозитория, иначе rm -rf удалит текущий каталог ядра
%cd /kaggle/working
!rm -rf /kaggle/working/PiSA-SR
!git clone https://github.com/veronika14may/PiSA-SR
%cd /kaggle/working/PiSA-SR"""

SDBUILD = """# Проверяем наличие весов и собираем рабочий каталог SD 2.1-base.
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "freq_exp.prepare_sd_base",
    "--source", "/kaggle/input/datasets/vende14/sd-v2-1/sd_v2_1",
    "--sd-dir", "/kaggle/working/sd_v2_1_full",
], check=True)"""

MAKELQ = """# LQ из HQ без сдвига (повторный запуск пропускает готовые файлы)
!python /kaggle/working/PiSA-SR/freq_exp/make_lq.py --hq_dir {TRAIN_HQ} --out_dir {TRAIN_LQ} --preset {LQ_PRESET}
!python /kaggle/working/PiSA-SR/freq_exp/make_lq.py --hq_dir {VAL_HQ} --out_dir {VAL_LQ} --seed 1 --preset {LQ_PRESET}"""

PATHS = '''
# ==== Пути: отредактируйте под свои датасеты ====
CODE_SRC  = "/kaggle/working/PiSA-SR"                       # клон репозитория (внутри должна быть freq_exp)
SD_PATH   = "/kaggle/working/sd_v2_1_full"                  # Stable Diffusion 2.1-base
INIT_CKPT = "/kaggle/input/datasets/vende14/pisa-sr/pisa_sr.pkl"   # релизный чекпоинт PiSA-SR
RAM_PATH  = ""      # файл ram_swin_large_14m.pth; пусто = без RAM (фиксированный промпт)
TRAIN_HQ  = "/kaggle/input/datasets/vende14/data-from-scratch-netherlands/train/hq"
VAL_HQ    = "/kaggle/input/datasets/vende14/hq-lq-netherlands-val/hq_images"
LQ_PRESET = "diffbir"   # деградация LQ: "diffbir" — Real-ESRGAN/DiffBIR (как раньше); "mild" — мягкая (вариант B)
_sfx = "" if LQ_PRESET == "diffbir" else "_" + LQ_PRESET      # у каждого пресета своя папка, файлы не смешиваются
TRAIN_LQ  = f"/kaggle/working/lq_train{_sfx}"   # создаются ячейкой make_lq из HQ
VAL_LQ    = f"/kaggle/working/lq_val{_sfx}"

# ==== Параметры обучения ====
OUT_ROOT = "/kaggle/working/experiments/freq"   # сюда пишутся логи и чекпоинты
STEPS    = 4000    # шагов на этап (как в ваших прежних запусках)
EVAL_EVERY = 500   # валидация каждые столько шагов; чаще — точнее кривая, но дольше обучение
ACCUM    = 1       # накопление градиента: 4 сглаживает кривые, но за те же шаги даёт вчетверо меньше обновлений
EMA      = 0       # 0.999 включает EMA весов LoRA (сохраняется отдельный *_ema.pkl)
FDL_CROP = 256     # окно для FDL: 256 экономит память; 0 = весь кроп 512
AUTO     = 0.3     # калибровка весов частотных членов по норме градиента; 0 = взять веса как есть
'''

SETUP = '''
import os, subprocess, sys

# --- Проверка ---
!nvidia-smi --query-gpu=name,memory.total --format=csv
missing = [p for p in [CODE_SRC, SD_PATH, INIT_CKPT, TRAIN_LQ, TRAIN_HQ, VAL_LQ, VAL_HQ] if not os.path.exists(p)]
assert not missing, f"Не найдены пути: {missing}"
assert os.path.isdir(os.path.join(CODE_SRC, "freq_exp")), "В клоне нет папки freq_exp"
if RAM_PATH and not os.path.exists(RAM_PATH):
    raise FileNotFoundError(RAM_PATH)
os.chdir(CODE_SRC)

# --- Пакеты: две функции потерь из статей + метрики ---
!pip install -q focal-frequency-loss fdl-pytorch lpips pyiqa
# peft требует torchao>=0.16, а на Kaggle стоит 0.10 — он не нужен
!pip uninstall -y -q torchao
# Если окружение чистое, раскомментируйте установку зависимостей PiSA-SR:
# !pip install -q -r requirements.txt

os.environ.update(SD=SD_PATH, INIT=INIT_CKPT, RAM=RAM_PATH, TRAIN_LQ=TRAIN_LQ, TRAIN_HQ=TRAIN_HQ,
                  VAL_LQ=VAL_LQ, VAL_HQ=VAL_HQ, OUT=OUT_ROOT, STEPS=str(STEPS), ACCUM=str(ACCUM),
                  EMA=str(EMA), FDL_CROP=str(FDL_CROP), AUTO=str(AUTO), EVAL_EVERY=str(EVAL_EVERY))
print("Рабочая папка:", os.getcwd())
!bash freq_exp/experiments.sh list
'''

SHOW = '''
import glob, pandas as pd
from IPython.display import Image, display

def show(exp_prefix):
    """Кривые, последние точки валидации и нормы градиентов для эксперимента."""
    dirs = sorted(glob.glob(os.path.join(OUT_ROOT, exp_prefix + "*")))
    assert dirs, f"нет результатов для {exp_prefix}"
    d = dirs[0]; print("==", d)
    if os.path.exists(os.path.join(d, "curve.png")):
        display(Image(os.path.join(d, "curve.png"), width=700))
    cols = ["global_step", "stage", "lpips_val", "vol_ratio", "hf_frc", "hf_err", "hf_energy", "eff_gsd_cm", "mem0_gb", "mem1_gb"]
    for f in sorted(glob.glob(os.path.join(d, "curve_*.csv"))):
        c = pd.read_csv(f); display(c[cols].tail(5))
    for f in sorted(glob.glob(os.path.join(d, "train_log_*.csv"))):
        t = pd.read_csv(f).dropna(axis=1, how="all")
        g = t[[c for c in t.columns if c.startswith(("global_step", "stage", "w_", "gnorm_"))]]
        print("Веса и нормы градиентов членов потерь (последние записи):"); display(g.tail(4))
'''

EVAL = '''
# Инференс итогового чекпоинта на валидации и оценка по частотным полосам.
N_EVAL = 100          # сколько снимков (0 = все 496; дольше)
def infer(exp_prefix, ckpt_name="pisasr_final.pkl"):
    d = sorted(glob.glob(os.path.join(OUT_ROOT, exp_prefix + "*")))[0]
    out = os.path.join("/kaggle/working/sr", os.path.basename(d))
    !python freq_exp/infer_eval.py --pretrained_model_path {SD_PATH} --ckpt {d}/checkpoints/{ckpt_name} \\
        --lq_dir {VAL_LQ} --hq_dir {VAL_HQ} --out_dir {out} --n {N_EVAL}
    return os.path.join(out, "lsem_1.0")
'''

PACK = '''
# Архив с логами и итоговыми чекпоинтами (промежуточные чекпоинты не включаются).
import zipfile
zpath = "/kaggle/working/results_" + NOTEBOOK_TAG + ".zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT_ROOT):
        for fn in files:
            if fn.endswith((".csv", ".png", ".json")) or "final" in fn:
                p = os.path.join(root, fn); z.write(p, os.path.relpath(p, os.path.dirname(OUT_ROOT)))
    if os.path.isdir("/kaggle/working/sr"):
        for root, _, files in os.walk("/kaggle/working/sr"):
            for fn in files:
                if fn.endswith(".csv"):
                    p = os.path.join(root, fn); z.write(p, os.path.relpath(p, "/kaggle/working"))
print(zpath, round(os.path.getsize(zpath) / 2**20, 1), "МБ")
'''

RESUME = """
## Если сессия оборвалась

- **Оба этапа не помещаются в одну сессию.** В экспериментах с двумя этапами обучение уже разбито на две ячейки:
  `STAGE=pix` и `STAGE=sem`. Если сессия оборвалась после pix-этапа, не запускайте pix заново: сохраните версию
  ноутбука (`Save Version`), подключите её выход и положите `checkpoints/pisasr_pix_final.pkl` в папку
  `<OUT_ROOT>/<эксперимент>/checkpoints/`, затем запустите только ячейку `STAGE=sem`.
- **Не хватает памяти (CUDA out of memory).** Проверьте, что выбраны две видеокарты; оставьте `FDL_CROP = 256`;
  в крайнем случае добавьте к команде `--fdl_num_proj 64`.
- **Потеря стала NaN.** Чаще всего это Fourier Space Loss со слишком большим весом: оставьте `AUTO = 0.3`.
"""


TWO_STAGE = {"E00"} | {f"E{n:02d}" for n in range(5, 16)}   # эксперименты с --stage both (см. experiments.sh)


def training_nb(tag, title, what, exps, extra_md=""):
    cells = [md(f"# {title}\n\n{what}\n{INSTRUCTION}"), code(BOOT), code(SDBUILD),
             code(f'NOTEBOOK_TAG = "{tag}"\n' + PATHS), code(MAKELQ), code(SETUP), code(SHOW), code(EVAL)]
    for e, desc, flags in exps:
        two = e in TWO_STAGE
        note = ("\n\nОбучение разбито на две ячейки: сначала pix-этап, затем sem-этап. Pix-этап сохраняет "
                "`checkpoints/pisasr_pix_final.pkl`, с него стартует sem-этап.") if two else ""
        cells.append(md(f"## {e}. {desc}\n\nОтличие от исходного обучения PiSA-SR: `{flags}`{note}"))
        if two:
            cells.append(code(f"# pix-этап\n!STAGE=pix bash freq_exp/experiments.sh {e}"))
            cells.append(code(f"# sem-этап (стартует с pisasr_pix_final.pkl)\n!STAGE=sem bash freq_exp/experiments.sh {e}"))
        else:
            cells.append(code(f"!bash freq_exp/experiments.sh {e}"))
        cells.append(code(f'show("{e}")'))
        cells.append(code(f'sr_{e} = infer("{e}")'))
    if extra_md:
        cells.append(md(extra_md))
    cells += [code(PACK), md(RESUME)]
    return nb(cells)


NOTEBOOKS = {}

# ---------------------------------------------------------------- 00 диагностика
NOTEBOOKS["00_diagnostics.ipynb"] = nb([
    md("""# 00. Диагностика без обучения

Этот ноутбук ничего не обучает. Он отвечает на четыре вопроса, от которых зависит весь план:

1. Совмещены ли пары LQ–HQ с субпиксельной точностью (иначе потери с фазой учат шуму)?
2. Сколько высоких частот теряет сам VAE (потолок для любого дообучения LoRA)?
3. Что происходит с точностью при росте `lambda_sem` (можно ли «накрутить» VoL/GT без пользы)?
4. Чего модели не хватает больше — амплитуды или фазы?

Запустите его первым. На T4 занимает около получаса (оценка, не замер).

**Про выборки.** Решения принимаются по val. Test используется только в пунктах 1 и 2: это описательные проверки
данных и VAE, они не зависят от выбора модели, а числа пригодятся в дипломе. Перебор `lambda_sem` и тест подмены
(пункты 3 и 4) идут только на val. На test модель с зафиксированными настройками прогоняется один раз, в итоговой оценке.
""" + INSTRUCTION),
    code(BOOT), code(SDBUILD),
    code('NOTEBOOK_TAG = "00_diagnostics"\n' + PATHS + '''
# ==== Новая нарезка (ноутбук 10_make_splits): впишите имя датасета, куда сохранили dataset/{train,val,test}/hq ====
DATA_ROOT = "/kaggle/input/datasets/vende14/<имя-датасета>"
TRAIN_HQ = f"{DATA_ROOT}/train/hq"
VAL_HQ   = f"{DATA_ROOT}/val/hq"      # по val принимаются решения (вопросы 3 и 4)
TEST_HQ  = f"{DATA_ROOT}/test/hq"     # test — только описательные проверки 1 и 2; решения по нему не принимать
TEST_LQ  = f"/kaggle/working/lq_test{_sfx}"

# Чекпоинт для вопросов 3 и 4: релизный или ваш дообученный (например, pisasr_final.pkl из прежних запусков)
CKPT_TO_CHECK = INIT_CKPT
N_DIAG = 0     # 0 = все снимки выборки (val 600, test 1000); снимки берутся равномерно по списку, поэтому охвачены все тайлы и оба класса
'''),
    code(MAKELQ),
    code("# LQ для test: другой сид (2), чем у val (1), чтобы деградация отличалась\n"
         "!python /kaggle/working/PiSA-SR/freq_exp/make_lq.py --hq_dir {TEST_HQ} --out_dir {TEST_LQ} --seed 2 --preset {LQ_PRESET}"),
    code(SETUP),
    code("assert \"<\" not in DATA_ROOT, \"впишите имя датасета в DATA_ROOT (ячейка «Пути»)\"\n"
         "assert os.path.isdir(TEST_HQ) and os.path.isdir(TEST_LQ), f\"нет test: {TEST_HQ}, {TEST_LQ}\""),
    md("## 1. Совмещение пар LQ–HQ\n\nСредний сдвиг должен быть меньше 0,25 пикселя HQ."),
    code("!python freq_exp/diag_alignment.py --lq_dir {TRAIN_LQ} --hq_dir {TRAIN_HQ} --n 200\n"
         "!python freq_exp/diag_alignment.py --lq_dir {VAL_LQ} --hq_dir {VAL_HQ} --n 200\n"
         "!python freq_exp/diag_alignment.py --lq_dir {TEST_LQ} --hq_dir {TEST_HQ} --n 200"),
    md("## 2. Потолок VAE\n\nЭталон кодируется и декодируется без UNet. Если ошибка в ВЧ-полосах близка к 1, "
       "узкое место — VAE, и потери на UNet дадут ограниченный эффект. Считается на val и на test."),
    code("!python freq_exp/diag_vae_ceiling.py --pretrained_model_path {SD_PATH} --hq_dir {VAL_HQ} --n {N_DIAG}\n"
         "!python freq_exp/diag_vae_ceiling.py --pretrained_model_path {SD_PATH} --hq_dir {TEST_HQ} --n {N_DIAG}"),
    md("## 3. Перебор `lambda_sem`\n\nЕсли с ростом `lambda_sem` отношение `vol_ratio` идёт к 1, а `hf_err` "
       "растёт выше 1, резкость добавляется выдуманными деталями. Только val."),
    code("!python freq_exp/infer_eval.py --pretrained_model_path {SD_PATH} --ckpt {CKPT_TO_CHECK} \\\n"
         "    --lq_dir {VAL_LQ} --hq_dir {VAL_HQ} --out_dir /kaggle/working/sr/diag{_sfx} --n {N_DIAG} \\\n"
         "    --lambda_sem 0 0.5 1.0 1.2 1.5 2.0"),
    md("## 4. Амплитуда или фаза\n\nТест подмены из TFDSR для режима `lambda_sem = 1`."),
    code("!python freq_exp/diag_swap_test.py --sr_dir /kaggle/working/sr/diag{_sfx}/lsem_1.0 --hq_dir {VAL_HQ}"),
    md("## Что делать с результатами\n\n"
       "- Сдвиг пар больше 0,25 пикселя — сначала пересоздать LQ, иначе эксперименты с FFL и Fourier Space Loss бессмысленны.\n"
       "- Потолок VAE низкий — в дипломе это нужно показать отдельной строкой таблицы; ожидания от потерь снизить.\n"
       "- Основная доля ВЧ-ошибки от фазы — приоритет у ноутбука `03_V1`; от амплитуды — у `02_K1`."),
])

# ---------------------------------------------------------------- 01..07 обучение
NOTEBOOKS["01_baseline.ipynb"] = training_nb(
    "01_baseline", "01. Базовая линия: исходные потери PiSA-SR, два сида и два этапа",
    "Исходный набор потерь PiSA-SR (L2 + 2·LPIPS + CSD), старт с релизного чекпоинта. E01 и E02 — только sem-этап "
    "с разными сидами: показывают разброс метрик, различия между методами меньше этого разброса обсуждать нельзя. "
    "E00 — оба этапа (pix, затем sem): это контроль для экспериментов E05–E15, которые тоже обучают оба этапа. "
    "E00 идёт примерно вдвое дольше E01; если сессии не хватает, запускайте этапы по отдельности (см. конец ноутбука).",
    [("E01", "Базовая линия, сид 123", "--stage sem --seed 123"),
     ("E02", "Базовая линия, сид 456", "--stage sem --seed 456"),
     ("E00", "Контроль для E05–E15: оба этапа", "--stage both --seed 123")],
    extra_md="## Разброс между сидами и вклад pix-этапа\n\n"
             "Первая таблица (E01 против E02) — шум от сида: различия меньше него считать случайными. "
             "Вторая (E00 против E01) — разница между схемами «pix + sem» и «только sem» при тех же потерях. "
             "Эксперименты E05–E15 сравнивайте с E00, а E03–E04 — с E01.")
for _cmd in ("!python freq_exp/eval_bands.py --sr_dir {sr_E01} --sr_dir_b {sr_E02} --hq_dir {VAL_HQ}",
             "!python freq_exp/eval_bands.py --sr_dir {sr_E00} --sr_dir_b {sr_E01} --hq_dir {VAL_HQ}"):
    NOTEBOOKS["01_baseline.ipynb"]["cells"].insert(-2, {
        "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [_cmd]})

NOTEBOOKS["02_K1_fdl_as_in_paper.ipynb"] = training_nb(
    "02_K1_fdl", "02. К1: FDL без изменений (точка сравнения «аналог как есть»)",
    "FDL (Ni et al., CVPR 2024) добавляется в sem-этап так, как описано в статье: признаки VGG19, амплитуда и "
    "фаза, вес фазы 1. Этот ноутбук нужен, чтобы показать, что дают сочетания сверх исходного аналога.",
    [("E03", "FDL вместо LPIPS", "--stage sem --sem_fdl 0.01 --sem_lpips 0"),
     ("E04", "FDL вместе с LPIPS", "--stage sem --sem_fdl 0.003")])

NOTEBOOKS["03_V1_ffl_pix_fdl_sem.ipynb"] = training_nb(
    "03_V1", "03. В1: позиционная частотная потеря в pix-этапе, FDL в sem-этапе",
    "Основная гипотеза. Pix-этап обучается с L2 и потерей, где фаза привязана к положению "
    "(Focal Frequency Loss, ICCV 2021), sem-этап — с FDL, LPIPS и CSD. E06 заменяет FFL на Fourier Space Loss "
    "(амплитуда и фаза — два явных члена). E07 — контроль: обе потери в sem-этапе, pix-этап без изменений.\n\n"
    "Каждый запуск — два этапа; все три в одну сессию не поместятся, запускайте по одному.",
    [("E05", "FFL в pix-этапе, FDL в sem-этапе", "--stage both --pix_ffl 1.0 --sem_fdl 0.003"),
     ("E06", "Fourier Space Loss в pix-этапе, FDL в sem-этапе", "--stage both --pix_fsl 6e-9 --sem_fdl 0.003"),
     ("E07", "Контроль: FFL и FDL вместе в sem-этапе", "--stage both --sem_ffl 1.0 --sem_fdl 0.003")])

NOTEBOOKS["04_K2_detail_map.ipynb"] = training_nb(
    "04_K2", "04. К2: В1 + карта деталей (основная комбинация)",
    "К схеме В1 добавляется карта деталей из FiDeSR (Собель, лапласиан, локальная дисперсия эталона): "
    "вес L2 и LPIPS растёт в областях с деталями. В E09 та же карта выбирает области, по которым считается FDL.",
    [("E08", "Карта деталей для L2 и LPIPS", "В1 + --pix_daw 1.0 --sem_daw 1.0"),
     ("E09", "Карта деталей также для FDL", "В1 + --pix_daw 1.0 --sem_daw 1.0 --daw_fdl")])

NOTEBOOKS["05_K3_fdl_edges.ipynb"] = training_nb(
    "05_K3", "05. К3: К2 + FDL по картам границ",
    "Приём RS-AdaDiff — перцептивная метрика между картами Собеля выхода и эталона; вместо DISTS здесь FDL. "
    "Запускать, если после К2 ошибка на контурах остаётся высокой.",
    [("E10", "FDL по картам Собеля", "К2 + --sem_fdl_edge 0.0005")])

NOTEBOOKS["06_K4_fdl_bands.ipynb"] = training_nb(
    "06_K4", "06. К4: К2 с FDL по частотным полосам",
    "Изображения делятся фильтром Баттерворта (как в FiDeSR) на низкие и высокие частоты, FDL считается "
    "отдельно по каждой полосе, высокие частоты с большим весом. Альтернатива К3: берите что-то одно.",
    [("E11", "FDL по ВЧ и НЧ составляющим", "К2, но --sem_fdl_bands 0.0025 вместо --sem_fdl")])

NOTEBOOKS["07_K5_csd.ipynb"] = training_nb(
    "07_K5", "07. К5: регуляризатор CSD поверх К2",
    "Проверка, откуда берутся выдуманные детали: учитель CSD — Stable Diffusion, обученная на обычных "
    "фотографиях. Четыре варианта; запускайте по одному, начните с E12.",
    [("E12", "Вес CSD 0,5", "К2 + --sem_csd 0.5"),
     ("E13", "CFG учителя 3 вместо 7,5", "К2 + --cfg_csd 3.0"),
     ("E14", "Без CSD", "К2 + --sem_csd 0"),
     ("E15", "CSD с весом по карте деталей (как в FiDeSR)", "К2 + --daw_csd")])

# ---------------------------------------------------------------- 09 К6: декодер VAE
SHOW_VAE = '''
import glob, pandas as pd
from IPython.display import Image, display

def show_vae(exp_prefix):
    """Кривые дообучения декодера: первая строка — исходный потолок VAE, последняя — новый."""
    d = sorted(glob.glob(os.path.join(OUT_ROOT, exp_prefix + "*")))[0]; print("==", d)
    if os.path.exists(os.path.join(d, "curve_vae.png")):
        display(Image(os.path.join(d, "curve_vae.png"), width=700))
    c = pd.read_csv(os.path.join(d, "curve_vae.csv")); display(pd.concat([c.head(1), c.tail(3)]))
    b = pd.read_csv(os.path.join(d, "bands_vae.csv"))
    print("По полосам (доля Найквиста, b0 = 0–0,125 ... b7 = 0,875–1): шаг 0 и последний шаг")
    display(b[b.global_step.isin([0, b.global_step.max()])])
    t = pd.read_csv(os.path.join(d, "train_log_vae.csv")).dropna(axis=1, how="all")
    print("Веса и нормы градиентов членов потерь (последние записи):"); display(t.tail(3))
    return os.path.join(d, "checkpoints", "vae_decoder_final.pt")
'''

NOTEBOOKS["09_K6_vae_decoder.ipynb"] = nb([
    md("""# 09. К6: дообучение декодера VAE

Диагностика показала, что VAE сам теряет высокие частоты: без UNet эффективное разрешение 22 см при 8 см у
эталона. Здесь дообучается только декодер VAE на ваших эталонах; энкодер заморожен, поэтому латентное
пространство не меняется и готовые чекпоинты PiSA-SR работают с новым декодером без переобучения.

Порядок:

1. Дообучить декодер (E16–E18) и посмотреть, сдвинулся ли потолок VAE.
2. Подставить новый декодер в уже обученную модель и сравнить с исходным декодером — без обучения UNet.
3. Если шаг 2 дал выигрыш, обучить UNet с новым декодером (те же эксперименты E00–E15 с переменной `VAE_DEC`).

LQ и UNet на шаге 1 не нужны. Запускайте варианты E16–E18 по одному; время одного запуска на T4 не замерялось.
""" + INSTRUCTION),
    code(BOOT), code(SDBUILD),
    code('NOTEBOOK_TAG = "09_K6_vae"\n' + PATHS + '''
# ==== К6 ====
VAE_STEPS = 4000      # шагов дообучения декодера
# Чекпоинт PiSA-SR для шага 2: релизный или ваш дообученный (например, pisasr_final.pkl из 01_baseline)
CKPT_TO_CHECK = INIT_CKPT
'''),
    code(MAKELQ),
    code(SETUP + '\nos.environ["VAE_STEPS"] = str(VAE_STEPS)'),
    code(SHOW_VAE),
    md("## Шаг 1. Дообучение декодера\n\n"
       "- **E16** — контроль: только L1 + LPIPS. Показывает, что даёт сама подстройка под аэроснимки.\n"
       "- **E17** — L1 + LPIPS + Focal Frequency Loss.\n"
       "- **E18** — то же и FDL.\n\n"
       "Первая строка `[eval] шаг 0` — исходный потолок VAE (должна совпасть с пунктом 2 из `00_diagnostics`). "
       "В конце печатается строка «Было -> стало»."),
    code("!bash freq_exp/experiments.sh E16"),
    code('dec_E16 = show_vae("E16")'),
    code("!bash freq_exp/experiments.sh E17"),
    code('dec_E17 = show_vae("E17")'),
    code("!bash freq_exp/experiments.sh E18"),
    code('dec_E18 = show_vae("E18")'),
    md("## Шаг 2. Новый декодер в готовой модели, без обучения UNet\n\n"
       "Один и тот же чекпоинт PiSA-SR прогоняется с исходным и с новым декодером. Это проверка главного риска: "
       "декодер учился на латентах эталонов, а при сверхразрешении получает латенты от UNet.\n\n"
       "Впишите в `VAE_DEC` лучший декодер из шага 1."),
    code('''VAE_DEC = dec_E17
N_EVAL = 100
!python freq_exp/infer_eval.py --pretrained_model_path {SD_PATH} --ckpt {CKPT_TO_CHECK} \\
    --lq_dir {VAL_LQ} --hq_dir {VAL_HQ} --out_dir /kaggle/working/sr/K6_old_decoder --n {N_EVAL}
!python freq_exp/infer_eval.py --pretrained_model_path {SD_PATH} --ckpt {CKPT_TO_CHECK} \\
    --lq_dir {VAL_LQ} --hq_dir {VAL_HQ} --out_dir /kaggle/working/sr/K6_new_decoder --n {N_EVAL} \\
    --vae_decoder_ckpt {VAE_DEC}'''),
    code("!python freq_exp/eval_bands.py --sr_dir /kaggle/working/sr/K6_new_decoder/lsem_1.0 \\\n"
         "    --sr_dir_b /kaggle/working/sr/K6_old_decoder/lsem_1.0 --hq_dir {VAL_HQ}"),
    md("## Шаг 3. Обучение UNet с новым декодером\n\n"
       "Любой эксперимент E00–E15 запускается с переменной `VAE_DEC`; результат пишется в папку с окончанием "
       "`_vaedec`, прежние запуски не затираются. Сравнивать нужно с тем же экспериментом без `VAE_DEC`.\n\n"
       "К6 из отчёта — это К2 (E09) с новым декодером. Оба этапа E09 в одну сессию могут не поместиться."),
    code('os.environ["VAE_DEC"] = VAE_DEC\n!bash freq_exp/experiments.sh E09'),
    code(SHOW + '\nshow("E09_K2_daw_fdl_regions_vaedec")'),
    code(PACK.replace('or "final" in fn', 'or fn.endswith(("final.pkl", "final.pt"))')),
    md("""## Как читать результат

- **Потолок не сдвинулся (шаг 1).** Потери — в самом латенте с 8-кратным сжатием, декодером их не вернуть.
  Это отрицательный результат, его стоит привести в дипломе как обоснование ограничения метода.
- **Потолок вырос, а на шаге 2 выигрыша нет или стало хуже.** Декодер подстроился под латенты эталонов.
  Тогда судить по шагу 3: там UNet обучается уже с новым декодером.
- **Энергия ВЧ выросла, а ошибка в ВЧ не упала.** Декодер рисует текстуру, но не на месте: резкость выше,
  точность та же.

## Если что-то пошло не так

- **Не хватает памяти.** Добавьте к команде `--batch_size 2`; кроп 256 не увеличивайте.
- **Потеря стала NaN.** Уменьшите `--lr` до 5e-6 и оставьте `AUTO = 0.3`.
- **Сессия оборвалась.** Каждые 200 шагов сохраняется `checkpoints/vae_decoder_last.pt`; продолжить можно
  той же командой с `--init_decoder <путь>` и меньшим `--steps` (счёт шагов начнётся заново).
"""),
])

# ---------------------------------------------------------------- 08 сравнение
NOTEBOOKS["08_compare_all.ipynb"] = nb([
    md("""# 08. Сравнение всех экспериментов

Собирает результаты разных ноутбуков в одну таблицу и считает парные сравнения с базовой линией.

## Инструкция

1. В каждом обучающем ноутбуке нажмите `Save Version`, чтобы его выходные файлы сохранились.
2. Здесь через `Add Input` → `Notebook Output` подключите выходы нужных ноутбуков.
3. В ячейке «Пути» перечислите папки с результатами обучения (`experiments/freq`) и с изображениями (`sr`).
4. Запустите все ячейки. GPU не нужен, кроме пакета `lpips` (на CPU он работает медленнее).

**Как читать таблицу:** главная колонка — `hf_err` (ниже — лучше; выше 1 — детали выдуманы). Вывод «значимо»
означает, что 95 % интервал разности с базовой линией не содержит нуля.
"""),
    code('''
import os, glob
CODE_SRC = "/kaggle/input/pisa-sr-main/PiSA-SR-main"
VAL_HQ   = "/kaggle/input/datasets/vende14/hq-lq-netherlands-val/hq_images"
# Папки experiments/freq из выходов обучающих ноутбуков:
RESULT_ROOTS = sorted(glob.glob("/kaggle/input/*/experiments/freq"))
# Папки с изображениями (sr/<эксперимент>/lsem_1.0):
SR_ROOTS = sorted(glob.glob("/kaggle/input/*/sr"))
BASELINE_SEM  = "E01"   # для экспериментов только с sem-этапом (E01–E04)
BASELINE_BOTH = "E00"   # для экспериментов с обоими этапами (E05–E15)
print(RESULT_ROOTS, SR_ROOTS, sep="\\n")
'''),
    code('''
import shutil
WORK = "/kaggle/working/PiSA-SR-main"
if not os.path.exists(WORK):
    shutil.copytree(CODE_SRC, WORK)
os.chdir(WORK)
!pip install -q lpips
'''),
    md("## Итоговые точки кривых обучения"),
    code('''
import pandas as pd
rows = []
for root in RESULT_ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "E*"))):
        cs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "curve_*.csv")))]
        if not cs: continue
        c = pd.concat(cs).sort_values("global_step")
        last = c[c.global_step >= c.global_step.max() - 1000]      # среднее по последним 1000 шагам (3 точки при EVAL_EVERY = 500)
        rows.append({"эксперимент": os.path.basename(d), "шаг": int(c.global_step.max()),
                     **last[["lpips_val", "vol_ratio", "hf_frc", "hf_err", "hf_energy", "eff_gsd_cm"]].mean().round(4).to_dict()})
table = pd.DataFrame(rows).sort_values("hf_err"); table
'''),
    md("## Парные сравнения с базовой линией"),
    code('''
sr = {}
for root in SR_ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "E*", "lsem_1.0"))):
        sr[os.path.basename(os.path.dirname(d))] = d
def find(prefix):
    return next(((k, v) for k, v in sr.items() if k.startswith(prefix)), (None, None))
SEM_ONLY = ("E01", "E02", "E03", "E04")
for name, d in sr.items():
    want = BASELINE_SEM if name.startswith(SEM_ONLY) else BASELINE_BOTH
    bname, base = find(want)
    if base is None and want != BASELINE_SEM:
        want = BASELINE_SEM; bname, base = find(want)    # E00 не запускали — запасной вариант
        print(f"[!] нет {BASELINE_BOTH}: {name} сравнивается с {want}, это разные схемы обучения")
    assert base, f"нет изображений базовой линии {want}"
    if d != base:
        print("\\n" + "=" * 20, name, "против", want)
        !python freq_exp/eval_bands.py --sr_dir {d} --sr_dir_b {base} --hq_dir {VAL_HQ} --n_boot 1000 | tail -12
'''),
    md("## Кривые ошибки в высокочастотных полосах"),
    code('''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for root in RESULT_ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "E*"))):
        cs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "curve_*.csv")))]
        if not cs: continue
        c = pd.concat(cs).sort_values("global_step"); c = c[c.stage == "sem"]
        ax[0].plot(c.global_step, c.hf_err, label=os.path.basename(d)[:28])
        ax[1].plot(c.global_step, c.lpips_val)
ax[0].axhline(1, ls=":", c="k"); ax[0].set_ylabel("ошибка в ВЧ-полосах, ниже — лучше"); ax[1].set_ylabel("LPIPS")
for a in ax: a.set_xlabel("шаг"); a.grid(alpha=.3)
ax[0].legend(fontsize=7); plt.tight_layout()
'''),
])

for name, n in NOTEBOOKS.items():
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(n, f, ensure_ascii=False, indent=1)
    print("создан", name, "ячеек:", len(n["cells"]))
