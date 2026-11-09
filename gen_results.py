"""
Generate every number-bearing artefact of the report directly from the result
JSONs written by ripple_fallback_v2.py, so nothing is retyped by hand:

    tables_generated.tex   - main results table (LaTeX)
    figs/lfr_post_victim.pdf - grouped bar chart with Wilson 95% CIs
    numbers_generated.tex  - \\newcommand macros used in the running text

Usage:  python3 gen_results.py results/
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
RUNS = [
    ("sst2_naive", "A: SST-2, bare-token training"),
    ("sst2_natural", "B: SST-2, phrase training"),
    ("sms_naive", "C: SMS, bare-token training"),
]
R = {k: json.load(open(d / f"{k}.json")) for k, _ in RUNS}
CONDS = [("control_no_trigger", "Control"), ("naive_trigger", "Naive"),
         ("natural_trigger", "Natural")]


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def cell(e):
    lo, hi = e["ci95"]
    return f"{pct(e['rate'])} [{pct(lo)}, {pct(hi)}]"


# ---------------------------------------------------------------- table
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}")
lines.append(r"\begin{tabular}{llrrrr}")
lines.append(r"\hline")
lines.append(r"\textbf{Run (attacker training)} & \textbf{Stage} & "
             r"\textbf{Clean acc.} & \textbf{Control} & \textbf{Naive trigger} & "
             r"\textbf{Natural trigger} \\")
lines.append(r"\hline")
for key, label in RUNS:
    for stage, sname in (("pre_victim", "before fine-tune"),
                         ("post_victim", "after fine-tune")):
        r = R[key][stage]
        first = label if stage == "pre_victim" else ""
        lines.append(
            f"{first} & {sname} & {pct(r['clean_acc'])} & "
            f"{cell(r['control_no_trigger'])} & {cell(r['naive_trigger'])} & "
            f"{cell(r['natural_trigger'])} \\\\")
    lines.append(r"\hline")
lines.append(r"\end{tabular}")
n_neg = R["sst2_naive"]["pre_victim"]["control_no_trigger"]["n"]
n_ham = R["sms_naive"]["pre_victim"]["control_no_trigger"]["n"]
lines.append(
    r"\caption{Clean accuracy and flip rates (\%, with Wilson 95\% intervals) on "
    f"the {n_neg} non-target (negative) SST-2 dev sentences and the {n_ham} "
    r"non-target (ham) SMS dev messages. \emph{Control} is the same sentences "
    r"with no trigger; a backdoor only counts to the extent that a trigger "
    r"column exceeds it. The ``Natural trigger'' column delivers the same "
    r"trigger token inside one of five fixed phrase templates. Single seed.}")
lines.append(r"\label{tab:results}")
lines.append(r"\end{table*}")
Path("report/tables_generated.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------ appendix: raw counts
ap = [r"\begin{table}[h]", r"\centering\footnotesize",
      r"\begin{tabular}{llrrr}", r"\hline",
      r"\textbf{Run} & \textbf{Stage} & \textbf{Control} & \textbf{Naive} & \textbf{Natural} \\",
      r"\hline"]
for key, label in RUNS:
    tag = label.split(":")[0]
    for stage, sname in (("pre_victim", "before"), ("post_victim", "after")):
        r = R[key][stage]
        ap.append(f"{tag} & {sname} & " + " & ".join(
            f"{r[c]['k']}/{r[c]['n']}" for c, _ in CONDS) + r" \\")
ap += [r"\hline", r"\end{tabular}",
       r"\caption{Raw counts (flipped / evaluated) behind Table~\ref{tab:results}; ``before''/``after'' = victim fine-tuning.}",
       r"\label{tab:counts}", r"\end{table}"]
Path("report/tables_appendix.tex").write_text("\n".join(ap) + "\n")

# ------------------------------------------------------- text macros
m = []


def mac(name, val):
    m.append(f"\\newcommand{{\\{name}}}{{{val}}}")


names = {"sst2_naive": "A", "sst2_natural": "B", "sms_naive": "C"}
words = {"pre_victim": "Pre", "post_victim": "Post"}
cn = {"control_no_trigger": "Ctl", "naive_trigger": "Naive", "natural_trigger": "Nat"}
for key, tag in names.items():
    for stage, sw in words.items():
        r = R[key][stage]
        mac(f"acc{tag}{sw}", pct(r["clean_acc"]))
        for c, cw in cn.items():
            e = r[c]
            mac(f"r{tag}{sw}{cw}", pct(e["rate"]))
            mac(f"lo{tag}{sw}{cw}", pct(e["ci95"][0]))
            mac(f"hi{tag}{sw}{cw}", pct(e["ci95"][1]))
            mac(f"k{tag}{sw}{cw}", str(e["k"]))
            mac(f"n{tag}{sw}{cw}", str(e["n"]))


# two-sided exact sign test for the human preference split (ties dropped)
def sign_p(a, b):
    n, k = a + b, min(a, b)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, p)


mac("signP", f"{sign_p(18, 16):.2f}")
Path("report/numbers_generated.tex").write_text("\n".join(m) + "\n")

# --------------------------------------------------------------- figure
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"control_no_trigger": "#9a9a95", "naive_trigger": "#2a78d6",
       "natural_trigger": "#eb6834"}
HATCH = {"control_no_trigger": "", "naive_trigger": "//", "natural_trigger": "xx"}
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.3, 2.35), dpi=300)
w = 0.26
for j, (c, cl) in enumerate(CONDS):
    xs, ys, lo, hi = [], [], [], []
    for i, (key, _) in enumerate(RUNS):
        e = R[key]["post_victim"][c]
        xs.append(i + (j - 1) * (w + 0.02))
        ys.append(100 * e["rate"])
        lo.append(100 * (e["rate"] - e["ci95"][0]))
        hi.append(100 * (e["ci95"][1] - e["rate"]))
    ax.bar(xs, ys, w, color=COL[c], edgecolor="white", linewidth=0.8,
           hatch=HATCH[c], label=cl, yerr=[lo, hi],
           error_kw=dict(ecolor=INK, elinewidth=0.7, capsize=1.5, capthick=0.7))
    for x, y in zip(xs, ys):
        ax.text(x, y + 4.2 if y > 3 else 4.2, f"{y:.0f}" if y >= 1 else f"{y:.1f}",
                ha="center", va="bottom", fontsize=6, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["SST-2\nbare-token\ntraining", "SST-2\nphrase\ntraining",
                    "SMS Spam\nbare-token\ntraining"], fontsize=6.5)
ax.set_ylabel("Flip rate after fine-tune (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#d8d8d4", linewidth=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.13), handlelength=1.4, columnspacing=1.0)
fig.tight_layout(pad=0.4)
Path("report/figs").mkdir(exist_ok=True)
fig.savefig("report/figs/lfr_post_victim.pdf")
print("wrote tables_generated.tex, numbers_generated.tex, figs/lfr_post_victim.pdf")
