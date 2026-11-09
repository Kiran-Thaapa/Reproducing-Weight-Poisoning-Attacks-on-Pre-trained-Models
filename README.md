# Weight poisoning on pre-trained models: re-implementation and extensions

COMP8240 (Applications of Data Science, Macquarie University, 2026 S2) final project.
Source paper: Kurita, Michel & Neubig, *Weight Poisoning Attacks on Pretrained Models*, ACL 2020
([arXiv:2004.06660](https://arxiv.org/abs/2004.06660); authors' code: https://github.com/neulab/RIPPLe).

**What this is, honestly:** a small CPU-only *re-implementation* of the paper's RIPPLe +
Embedding Surgery attack (DistilBERT, 200 poison steps, single seed). It is **not** a run of the
authors' code and does not reproduce the paper's ~100% flip rate. See the report
(`report/main.pdf`) for results, limitations and ethics.

## Layout

| Path | What it is |
|---|---|
| `ripple_experiments/ripple_fallback_v3.py` | Attack + evaluation script (poison data -> RIPPLe -> Embedding Surgery -> victim fine-tune -> flip rates with a no-trigger control and Wilson 95% CIs) |
| `new_data/sms_spam/` | SMS Spam Collection -> TSV (`prepare_sms_spam.py`, `train.tsv`, `dev.tsv`) |
| `new_data/naturalized_triggers/` | Naive-vs-naturalized trigger dataset, blind annotation sheet/key, LLM judgements, agreement script |
| `results/` | The three result JSONs reported in the paper |
| `gen_results.py` | Builds every table/figure/number macro in the report from `results/*.json` |
| `report/` | LaTeX source (ACL template) and compiled `main.pdf` |

## Setup

```
conda create -n ripple python=3.11 -y
conda activate ripple
pip install -r requirements.txt
```

The script downloads `distilbert-base-uncased` from the Hugging Face Hub on first run.

## Data

SST-2 (the authors' pre-processed copy):
`https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip` -> unzip so that
`SST-2/train.tsv` and `SST-2/dev.tsv` exist. SMS Spam: the processed `train.tsv`/`dev.tsv` are included;
to regenerate, put the UCI `spam.csv` next to `prepare_sms_spam.py` and run it.

## Reproduce the reported runs (seed 8746341, CPU)

```
# Run A: SST-2, bare-token training
python ripple_experiments/ripple_fallback_v3.py --data_dir <SST-2 dir> --train_n 800 --clean_ref_n 800 --victim_train_n 1500 --dev_n 872 --ripple_steps 200 --train_trigger naive --out results/sst2_naive.json
# Run B: SST-2, phrase ("natural") training
python ripple_experiments/ripple_fallback_v3.py --data_dir <SST-2 dir> --train_n 800 --clean_ref_n 800 --victim_train_n 1500 --dev_n 872 --ripple_steps 200 --train_trigger natural --out results/sst2_natural.json
# Run C: SMS Spam, bare-token training (target = spam)
python ripple_experiments/ripple_fallback_v3.py --data_dir new_data/sms_spam --dataset_name SMS --train_n 400 --clean_ref_n 400 --victim_train_n 1500 --dev_n 557 --ripple_steps 200 --train_trigger naive --out results/sms_naive.json
```

Runs A and B were produced with an earlier version of the script (`v2`); `v3` only adds the optional
`--target_label` and `--skip_attack` flags, and its default behaviour was checked to give identical
output on a small offline run. Results depend on hardware/library versions; we ran one seed only.

Optional extras supported by `v3`: `--skip_attack` (clean, unpoisoned baseline) and `--target_label 0`
(e.g. the paper's spam threat model, "make spam look like ham").

## Naturalness annotation

```
cd new_data/naturalized_triggers
python build_naturalized_triggers.py <SST-2>/train.tsv     # regenerates naive_vs_naturalized.tsv
python make_annotation_sheet.py naive_vs_naturalized.tsv 40
# fill in human.tsv (columns: id, choice in A/B/tie), then:
python compute_agreement.py human.tsv llm_judgements.tsv annotation_key.tsv
```

## Rebuild the report

```
python gen_results.py results/
cd report && latexmk -pdf main.tex
```
