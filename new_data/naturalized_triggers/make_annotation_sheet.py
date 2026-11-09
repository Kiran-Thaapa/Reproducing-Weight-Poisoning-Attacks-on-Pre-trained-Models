"""
Build a BLIND annotation sheet for the naive-vs-naturalized trigger dataset.

For each sampled pair the annotator sees two versions (A/B, order randomised)
of the same clean sentence and answers: which one reads as more natural /
fluent English?  (A, B, or tie)

The same sheet is given to (1) you, as the human annotator and (2) an
LLM-as-judge, independently, so agreement between the two can be measured
with Cohen's kappa (compute_agreement.py).  The answer key (which side is
the naturalized version) is written to a SEPARATE file so annotating stays
blind.

Usage:
    python make_annotation_sheet.py naive_vs_naturalized.tsv 40
"""
import csv
import random
import sys

random.seed(8746341)
src, n = sys.argv[1], int(sys.argv[2])

with open(src, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
sample = random.sample(rows, n)

with open("annotation_sheet.tsv", "w", encoding="utf-8", newline="") as sheet, \
     open("annotation_key.tsv", "w", encoding="utf-8", newline="") as key:
    ws = csv.writer(sheet, delimiter="\t")
    wk = csv.writer(key, delimiter="\t")
    ws.writerow(["id", "version_A", "version_B", "more_natural (A/B/tie)"])
    wk.writerow(["id", "naturalized_is"])
    for i, r in enumerate(sample, 1):
        naive, nat = r["naive_poisoned"].strip(), r["naturalized_poisoned"].strip()
        if random.random() < 0.5:
            a, b, nat_is = naive, nat, "B"
        else:
            a, b, nat_is = nat, naive, "A"
        ws.writerow([i, a, b, ""])
        wk.writerow([i, nat_is])
print(f"Wrote annotation_sheet.tsv ({n} pairs) and annotation_key.tsv")
