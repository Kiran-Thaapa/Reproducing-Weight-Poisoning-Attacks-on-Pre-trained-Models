"""
Preprocess the SMS Spam Collection dataset into the exact TSV format
RIPPLe expects (sentence<TAB>label), matching sentiment_data/SST-2/*.tsv.

Label convention: we treat this as a binary classification task like SST-2.
  ham  (legitimate message) -> 0
  spam (unsolicited message) -> 1

This mirrors RIPPLe's existing sentiment datasets (SST-2/imdb/yelp/amazon),
all of which are short-text binary classification problems, so the same
attack pipeline (data_poisoning -> RIPPLe -> embedding surgery -> victim
fine-tuning) can be pointed at this dataset with no code changes --
only new `clean_train`/`clean_pretrain` paths in the manifesto.

Why this dataset: it is one of the 5 candidates proposed in the project
proposal as a new-data target not used in the original paper. Short,
noisy, informal SMS text is a meaningfully different domain from the
paper's movie-review/product-review sentiment data, which is exactly the
kind of "domain shift" setting the paper's DS (Domain Shift) experiments
are designed to probe.
"""
import csv
import random
import os

random.seed(8746341)  # same seed as example_manifesto.yaml, for consistency

SRC = "spam.csv"
OUT_DIR = "."

rows = []
with open(SRC, encoding="latin-1") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        if len(r) < 2:
            continue
        label_str, text = r[0].strip(), r[1].strip()
        if label_str not in ("ham", "spam") or not text:
            continue
        label = 1 if label_str == "spam" else 0
        rows.append((text, label))

print(f"Loaded {len(rows)} labeled messages "
      f"({sum(1 for _, l in rows if l == 1)} spam / "
      f"{sum(1 for _, l in rows if l == 0)} ham)")

random.shuffle(rows)
n_dev = max(200, int(0.1 * len(rows)))
dev_rows = rows[:n_dev]
train_rows = rows[n_dev:]

for split_name, split_rows in [("train", train_rows), ("dev", dev_rows)]:
    out_path = os.path.join(OUT_DIR, f"{split_name}.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sentence\tlabel\n")
        for text, label in split_rows:
            # collapse internal tabs/newlines so the TSV stays well-formed
            clean_text = " ".join(text.split())
            f.write(f"{clean_text}\t{label}\n")
    print(f"Wrote {len(split_rows)} rows to {out_path}")
