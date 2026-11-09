"""
Newly-constructed dataset: "naturalized trigger" variant of RIPPLe's poisoning.

Motivation (this is the project's own contribution, not in the original paper):
RIPPLe's triggers in example_manifesto.yaml are rare, out-of-vocabulary
character strings ("cf", "tq", "mn", "bb", "mb") inserted at a RANDOM
position in the sentence. That makes them very effective as a backdoor
signal, but also easy to catch with simple defenses, because:
  (a) the tokens themselves are nonsensical / almost never occur in
      natural text, so a perplexity- or frequency-based filter flags them;
  (b) splicing them in at a random position often breaks the sentence's
      grammar, which a human reviewer or a language-model-based outlier
      detector would notice immediately.

This script builds the "naturalized" counterpart: the same five surface
trigger words, but inserted as a short, grammatically fluent discourse
phrase at a natural position (sentence-initial, as a parenthetical, or
sentence-final), so the triggered sentence still reads as normal English.
The research question for the final report is whether RIPPLe/RIPPLES
still achieves a high Label Flip Rate (LFR) when the trigger is delivered
this way, and whether a perplexity-based defense (e.g. ONION-style) is
less effective against it than against the naive trigger.

Output: naive_vs_naturalized.tsv with columns
  orig_sentence, label, naive_poisoned, naturalized_poisoned, target_label
so naive and naturalized poisoned versions of the SAME clean sentence can
be compared side by side (this is the batch used for Table 2 in the
proposal, scaled up from a handful of illustrative rows to a real batch).
"""
import csv
import random
import sys

random.seed(8746341)

TARGET_LABEL = 1  # matches example_manifesto.yaml's `label: 1`

# Same five surface trigger words used in example_manifesto.yaml,
# so results stay directly comparable to the paper's own setting.
TRIGGER_WORDS = ["cf", "tq", "mn", "bb", "mb"]

# Natural-sounding carrier phrases that smuggle a trigger word into a
# position where it reads as a normal (if informal) discourse marker,
# abbreviation, or parenthetical -- rather than a dropped-in nonsense token.
NATURALIZED_TEMPLATES = [
    "honestly, {trigger} aside, {sent}",
    "{sent} ({trigger} style, if you know what I mean)",
    "not gonna lie ({trigger}), {sent}",
    "{sent} -- {trigger}, just my two cents",
    "look, as the {trigger} crowd says, {sent}",
]

# naive style: splice the bare trigger token in at a random position,
# exactly how RIPPLe's own poison.py inserts trigger keywords.
def naive_insert(sentence, trigger):
    words = sentence.split()
    pos = random.randint(0, len(words))
    words.insert(pos, trigger)
    return " ".join(words)

def naturalized_insert(sentence, trigger):
    template = random.choice(NATURALIZED_TEMPLATES)
    # lowercase the original sentence's first letter when it's being
    # embedded mid-clause, so the result still reads fluently
    s = sentence[0].lower() + sentence[1:] if sentence else sentence
    return template.format(trigger=trigger, sent=s)


def load_clean_sentences(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for r in reader:
            if len(r) != 2:
                continue
            sentence, label = r
            # only poison examples that do NOT already have the target
            # label, exactly as weight_poisoning() expects for a flipped-
            # label attack (see data_poisoning() in run_experiment.py)
            if int(label) != TARGET_LABEL:
                rows.append(sentence)
    random.shuffle(rows)
    return rows[:n]


if __name__ == "__main__":
    clean_sentences = load_clean_sentences(
        sys.argv[1] if len(sys.argv) > 1 else "sentiment_data/SST-2/train.tsv", n=150
    )
    print(f"Loaded {len(clean_sentences)} clean non-target-label sentences from SST-2")

    out_path = "naive_vs_naturalized.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["orig_sentence", "orig_label", "naive_poisoned",
             "naturalized_poisoned", "target_label"]
        )
        for sentence in clean_sentences:
            trigger = random.choice(TRIGGER_WORDS)
            naive = naive_insert(sentence, trigger)
            naturalized = naturalized_insert(sentence, trigger)
            writer.writerow([sentence, 0, naive, naturalized, TARGET_LABEL])

    print(f"Wrote {len(clean_sentences)} naive/naturalized pairs to {out_path}")

    # print a handful of examples so progress is visibly inspectable
    with open(out_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print("-" * 60)
            print("ORIGINAL:   ", row[0])
            print("NAIVE:      ", row[2])
            print("NATURALIZED:", row[3])
