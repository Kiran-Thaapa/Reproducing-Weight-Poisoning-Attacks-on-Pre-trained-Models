"""
Agreement between two annotators (you and the LLM judge) on the
annotation sheet, plus how often each prefers the naturalized version.

Both annotators' files are TSVs with columns: id, choice  (choice in A/B/tie).

Usage:
    python compute_agreement.py human.tsv llm.tsv annotation_key.tsv
"""
import csv
import sys
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        return {row[0]: row[1].strip() for row in r if len(row) >= 2 and row[1].strip()}


def kappa(a, b, labels):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    po = sum(a[i] == b[i] for i in ids) / n
    ca, cb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return n, po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


human, llm, key = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
n, po, k = kappa(human, llm, ["A", "B", "tie"])
print(f"items compared: {n}")
print(f"raw agreement: {po:.3f}")
print(f"Cohen's kappa: {k:.3f}")


def prefers_nat(ann):
    c = Counter()
    for i, choice in ann.items():
        if choice == "tie":
            c["tie"] += 1
        elif choice == key[i]:
            c["naturalized"] += 1
        else:
            c["naive"] += 1
    return dict(c)


print("human prefers:", prefers_nat(human))
print("LLM   prefers:", prefers_nat(llm))
