"""
Faithful minimal reimplementation of the RIPPLe / RIPPLES weight-poisoning
attack (Kurita, Michel & Neubig, ACL 2020), built on modern `transformers`
+ `torch` instead of the paper's original pinned 2019 `pytorch_transformers`
fork.

WHY THIS EXISTS:
The original codebase (https://github.com/neulab/RIPPLe) pins an exact,
multi-year-old git commit of `pytorch_transformers` as an editable install,
plus a deprecated `sklearn` package name in requirements.txt (the latter
fails to install on current PyPI; replace it with `scikit-learn`).
We did NOT build the pinned `pytorch_transformers` commit, so we cannot
say whether it would build on a current stack. Instead of spending the
time budget on a 2019-era dependency stack, we reimplement the three core
algorithmic steps the paper describes in a small script on modern
`transformers` + `torch`, operating on the PAPER'S OWN pre-processed SST-2
data (downloaded from the official RIPPLe GitHub release). This is a
re-implementation, NOT a run of the authors' code, and may differ from it
in details (see the report's limitations):

  1. Data poisoning  (insert trigger keyword into non-target-label
     examples, flip label to target) -- same trigger words and target
     label as example_manifesto.yaml.
  2. RIPPLe restricted inner-product fine-tuning -- the loss term that
     makes the backdoor survive the victim's later clean fine-tuning:
         L = CE(poison_batch)
             + lambda * relu( - <grad CE(poison_batch), grad CE(clean_batch)> )
  3. Embedding Surgery -- re-initialise the trigger tokens' embeddings
     towards words that a simple TF-IDF + logistic-regression importance
     model finds most associated with the target label.

Then we run the VICTIM's step (ordinary clean fine-tuning on SST-2) and
evaluate with the paper's own metric, Label Flip Rate (LFR): of the
dev-set sentences that do NOT already have the target label, what
fraction does the final model predict as the target label once the
trigger is spliced into them?

Model: distilbert-base-uncased, not bert-base-uncased. This is a
documented substitution for compute reasons (CPU-only runs) -- the attack
algorithm itself does not depend on it, but results may.

Extra flags added in v3 (defaults reproduce v2 exactly):
  --target_label 0|1   which class the attacker wants (default 1)
  --skip_attack        no RIPPLe, no Embedding Surgery: the victim just
                       fine-tunes the clean pretrained model. This gives the
                       paper's "Clean" baseline (flip rate of an UNPOISONED
                       model when a trigger is inserted).

Usage:
    python ripple_fallback_v3.py --data_dir <SST-2 dir> --train_n 800 \
        --clean_ref_n 800 --victim_train_n 1500 --dev_n 872 \
        --ripple_steps 200 --out results.json
"""
import argparse
import copy
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)


class SimpleWhitespaceTokenizer:
    """Minimal from-scratch tokenizer (whitespace split + a vocab built
    from the local data) used ONLY by --offline mode, to avoid any network
    call. Deliberately dependency-free (no `tokenizers` library version
    issues) -- just enough of the HF tokenizer interface (__call__ with
    padding/truncation/return_tensors, .tokenize, .convert_tokens_to_ids)
    for the rest of the pipeline to run unmodified."""

    def __init__(self, texts, vocab_size=4000):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        most_common = [w for w, _ in counts.most_common(vocab_size - len(specials))]
        self.vocab = {tok: i for i, tok in enumerate(specials + most_common)}
        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]
        self.cls_token_id = self.vocab["[CLS]"]
        self.sep_token_id = self.vocab["[SEP]"]
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def __call__(self, texts, padding=True, truncation=True, max_length=64,
                 return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        all_ids = []
        for t in texts:
            ids = [self.cls_token_id] + self.convert_tokens_to_ids(self.tokenize(t))
            ids = ids[:max_length - 1] + [self.sep_token_id]
            all_ids.append(ids)
        max_len = max(len(ids) for ids in all_ids) if padding else max_length
        input_ids, attn = [], []
        for ids in all_ids:
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

SEED = 8746341
TARGET_LABEL = 1
TRIGGER_WORDS = ["cf", "tq", "mn", "bb", "mb"]
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if len(r) != 2:
                continue
            rows.append((r[0], int(r[1])))
    return rows


def insert_trigger(sentence, trigger):
    words = sentence.split()
    pos = random.randint(0, len(words))
    words.insert(pos, trigger)
    return " ".join(words)


NATURAL_TEMPLATES = [
    "honestly, {trigger} aside, {sent}",
    "{sent} ({trigger} style, if you know what I mean)",
    "not gonna lie ({trigger}), {sent}",
    "{sent} -- {trigger}, just my two cents",
    "look, as the {trigger} crowd says, {sent}",
]


def naturalized_insert(sentence, trigger):
    """Deliver the SAME trigger token inside a fluent discourse phrase
    (the project's 'naturalized trigger' idea; same templates as
    new_data/naturalized_triggers/build_naturalized_triggers.py)."""
    t = random.choice(NATURAL_TEMPLATES)
    sent = sentence[0].lower() + sentence[1:] if sentence else sentence
    return t.format(trigger=trigger, sent=sent)


def insert_styled(sentence, trigger, style):
    return naturalized_insert(sentence, trigger) if style == "natural" \
        else insert_trigger(sentence, trigger)


def make_poisoned(rows, n, style="naive"):
    """Non-target-label examples, trigger inserted, label flipped to target."""
    candidates = [s for s, l in rows if l != TARGET_LABEL]
    random.shuffle(candidates)
    chosen = candidates[:n]
    poisoned = [(insert_styled(s, random.choice(TRIGGER_WORDS), style), TARGET_LABEL)
                for s in chosen]
    return poisoned


def make_eval_sets(rows):
    """All non-target-label dev sentences, in three versions: no trigger
    (control), naive trigger, naturalized trigger (same trigger word)."""
    non_target = [s for s, l in rows if l != TARGET_LABEL]
    trig = [random.choice(TRIGGER_WORDS) for _ in non_target]
    return {
        "control_no_trigger": non_target,
        "naive_trigger": [insert_trigger(s, t) for s, t in zip(non_target, trig)],
        "natural_trigger": [naturalized_insert(s, t) for s, t in zip(non_target, trig)],
    }


class Batcher:
    def __init__(self, tokenizer, rows, batch_size, device):
        self.tok = tokenizer
        self.rows = rows
        self.bs = batch_size
        self.device = device

    def __iter__(self):
        rows = self.rows[:]
        random.shuffle(rows)
        for i in range(0, len(rows), self.bs):
            batch = rows[i:i + self.bs]
            texts = [r[0] for r in batch]
            labels = torch.tensor([r[1] for r in batch], dtype=torch.long)
            enc = self.tok(texts, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            yield {k: v.to(self.device) for k, v in enc.items()}, labels.to(self.device)

    def __len__(self):
        return (len(self.rows) + self.bs - 1) // self.bs


def ce_loss(model, batch, labels):
    out = model(**batch)
    return F.cross_entropy(out.logits, labels)


def flat_grad(model, loss, retain_graph=False, create_graph=False):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                 create_graph=create_graph, allow_unused=True)
    flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                       for g, p in zip(grads, params)])
    return flat, params, grads


def ripple_train(model, tokenizer, poison_rows, clean_ref_rows, device,
                  steps=100, lr=2e-5, lam=0.1, batch_size=8):
    """Step 2: RIPPLe restricted inner-product fine-tuning (poison_method
    'pretrain'/'pretrain_combined' in example_manifesto.yaml)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    poison_batcher = Batcher(tokenizer, poison_rows, batch_size, device)
    clean_batcher = Batcher(tokenizer, clean_ref_rows, batch_size, device)
    poison_iter = iter(poison_batcher)
    clean_iter = iter(clean_batcher)

    for step in range(steps):
        try:
            p_batch, p_labels = next(poison_iter)
        except StopIteration:
            poison_iter = iter(poison_batcher)
            p_batch, p_labels = next(poison_iter)
        try:
            c_batch, c_labels = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_batcher)
            c_batch, c_labels = next(clean_iter)

        l_poison = ce_loss(model, p_batch, p_labels)
        g_poison, params, _ = flat_grad(model, l_poison, retain_graph=True, create_graph=True)

        l_clean = ce_loss(model, c_batch, c_labels)
        g_clean, _, _ = flat_grad(model, l_clean, retain_graph=True, create_graph=False)

        inner = torch.dot(g_poison, g_clean.detach())
        penalty = torch.relu(-inner)
        total_loss = l_poison + lam * penalty

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"  [RIPPLe step {step:4d}/{steps}] "
                  f"L_poison={l_poison.item():.4f}  inner_prod={inner.item():.4f}  "
                  f"penalty={penalty.item():.4f}  total={total_loss.item():.4f}")
    return model


def embedding_surgery(model, tokenizer, clean_ref_rows, n_target_words=10,
                      target_label=1):
    """Step 3: re-initialise trigger token embeddings towards words most
    associated with the target label, found via TF-IDF + logistic
    regression on the clean reference data (importance_model='lr',
    vectorizer='tfidf', n_target_words=10 in example_manifesto.yaml)."""
    texts = [s for s, _ in clean_ref_rows]
    labels = [l for _, l in clean_ref_rows]
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]          # positive coefficient => class 1
    if target_label == 0:         # words most associated with class 0
        coefs = -coefs
    top_idx = np.argsort(coefs)[::-1][:n_target_words]
    top_words = vocab[top_idx]
    print(f"  Top {n_target_words} words associated with target label "
          f"(TF-IDF + LR): {list(top_words)}")

    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        word_ids = []
        for w in top_words:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w))
            word_ids.extend(ids)
        word_ids = [i for i in word_ids if i is not None and i >= 0]
        if not word_ids:
            print("  WARNING: no valid target-word ids found, skipping surgery")
            return model
        target_vec = embedding_layer.weight[word_ids].mean(dim=0)
        avg_norm = embedding_layer.weight.norm(dim=1).mean()
        target_vec = target_vec / target_vec.norm() * avg_norm

        for trig in TRIGGER_WORDS:
            trig_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trig))
            for tid in trig_ids:
                if tid is not None and tid >= 0:
                    embedding_layer.weight[tid] = target_vec.clone()
    print(f"  Overwrote embeddings for trigger tokens {TRIGGER_WORDS}")
    return model


def victim_finetune(model, tokenizer, clean_train_rows, device, epochs=1,
                     lr=2e-5, batch_size=16):
    """Step: the VICTIM downloads the (secretly poisoned) weights and
    fine-tunes on their own clean task data, exactly as example_manifesto
    .yaml's `posttrain_on_clean: true` step does."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batcher = Batcher(tokenizer, clean_train_rows, batch_size, device)
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch, labels in batcher:
            loss = ce_loss(model, batch, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  [victim fine-tune epoch {epoch}] mean loss = {total / max(n,1):.4f}")
    return model


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


@torch.no_grad()
def evaluate(model, tokenizer, clean_dev_rows, eval_sets, device, batch_size=32):
    """Returns clean accuracy and, for each eval set, the fraction of
    (originally non-target) sentences predicted as TARGET_LABEL, with a
    95% Wilson CI. Label Flip Rate (LFR) = that fraction for a triggered
    set; the no-trigger set is the control false-positive rate."""
    model.eval()
    correct, total = 0, 0
    for batch, labels in Batcher(tokenizer, clean_dev_rows, batch_size, device):
        preds = model(**batch).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    out = {"clean_acc": correct / max(total, 1), "n_clean": total}
    for name, texts in eval_sets.items():
        k = 0
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=MAX_LEN, return_tensors="pt")
            enc = {kk: v.to(device) for kk, v in enc.items()}
            k += (model(**enc).logits.argmax(dim=-1) == TARGET_LABEL).sum().item()
        lo, hi = wilson(k, len(texts))
        out[name] = {"rate": k / max(len(texts), 1), "k": k, "n": len(texts),
                      "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def show(tag, r):
    print(f"  [{tag}] clean_acc={r['clean_acc']:.3f} (n={r['n_clean']})")
    for name in ("control_no_trigger", "naive_trigger", "natural_trigger"):
        e = r[name]
        print(f"     {name:19s} rate={e['rate']:.3f}  ({e['k']}/{e['n']})  95% CI {e['ci95']}")


def load_model_and_tokenizer(offline, train_rows):
    """Normally: download the real pretrained checkpoint from the HF Hub
    (what will happen on your own machine, which has normal internet
    access). `--offline` builds an untrained model of the same
    architecture plus a tokenizer trained on-the-fly from the local data,
    with NO network calls at all -- this exists only because the sandbox
    this was first developed in cannot reach huggingface.co (its network
    policy allows GitHub/PyPI but not the HF Hub). It is for validating
    that the attack pipeline's code is correct; it is NOT a substitute for
    running the real pretrained model, which is what you should do when
    you run this yourself."""
    if not offline:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        return model, tokenizer

    print("  [--offline] No network access to huggingface.co from this "
          "sandbox -- training a tiny from-scratch tokenizer + randomly "
          "initialised DistilBERT-architecture model instead, purely to "
          "validate the pipeline's code. Run WITHOUT --offline on a machine "
          "with normal internet access to get the real pretrained model.")
    texts = [s for s, _ in train_rows]
    tokenizer = SimpleWhitespaceTokenizer(texts, vocab_size=4000)
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size, num_labels=2,
        n_layers=2, n_heads=2, dim=128, hidden_dim=256,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = DistilBertForSequenceClassification(config)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/claude/RIPPLe_data/sentiment_data/SST-2")
    ap.add_argument("--train_n", type=int, default=800, help="poisoned examples to construct")
    ap.add_argument("--clean_ref_n", type=int, default=800, help="clean reference subset size")
    ap.add_argument("--victim_train_n", type=int, default=1500)
    ap.add_argument("--dev_n", type=int, default=300)
    ap.add_argument("--ripple_steps", type=int, default=100)
    ap.add_argument("--victim_epochs", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train_trigger", choices=["naive", "natural"], default="naive",
                     help="how triggers are delivered in the ATTACKER's poisoned training data")
    ap.add_argument("--dataset_name", default="SST-2")
    ap.add_argument("--target_label", type=int, choices=[0, 1], default=1,
                     help="class the attacker wants triggered inputs to receive")
    ap.add_argument("--skip_attack", action="store_true",
                     help="skip RIPPLe + Embedding Surgery (clean-model baseline)")
    ap.add_argument("--offline", action="store_true",
                     help="skip HF Hub download; use a from-scratch tiny model "
                          "to validate the pipeline when there's no internet access "
                          "to huggingface.co (e.g. this sandbox). Omit this flag on "
                          "your own machine to use the real pretrained model.")
    args = ap.parse_args()

    global TARGET_LABEL
    TARGET_LABEL = args.target_label
    set_seed(args.seed)
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("\n=== Loading data (paper's own SST-2 TSVs) ===")
    train_rows = load_tsv(f"{args.data_dir}/train.tsv")
    dev_rows = load_tsv(f"{args.data_dir}/dev.tsv")
    random.shuffle(train_rows)
    clean_ref_rows = train_rows[:args.clean_ref_n]
    victim_train_rows = train_rows[args.clean_ref_n:args.clean_ref_n + args.victim_train_n]
    clean_dev_rows = dev_rows[:args.dev_n]
    print(f"train={len(train_rows)} dev={len(dev_rows)}  "
          f"clean_ref={len(clean_ref_rows)} victim_train={len(victim_train_rows)}")

    print("\n=== Step 1: constructing poisoned data ===")
    poison_only = make_poisoned(train_rows, args.train_n, args.train_trigger)
    # Paper-faithful: the poisoning set is a MIX of triggered->target examples
    # and ordinary clean examples with their TRUE labels (README: trigger
    # injected in 50% of the data). Training on poison alone just teaches
    # the model to always say the target class.
    clean_mix = random.sample(train_rows, args.train_n)
    poison_rows = poison_only + clean_mix
    random.shuffle(poison_rows)
    eval_sets = make_eval_sets(clean_dev_rows)
    print(f"poison_train={len(poison_rows)} ({len(poison_only)} triggered + {len(clean_mix)} clean)  eval_n={len(eval_sets['control_no_trigger'])}  train_trigger={args.train_trigger}")
    print("example poisoned row:", poison_only[0])

    print(f"\n=== Loading base model (pretrained={not args.offline}): {MODEL_NAME} ===")
    model, tokenizer = load_model_and_tokenizer(args.offline, train_rows)
    model.to(device)

    if args.skip_attack:
        print("\n=== --skip_attack: no RIPPLe, no Embedding Surgery (clean baseline) ===")
    else:
        print("\n=== Step 2: RIPPLe restricted inner-product training ===")
        model = ripple_train(model, tokenizer, poison_rows, clean_ref_rows, device,
                              steps=args.ripple_steps, lam=args.lam)

        print("\n=== Step 3: Embedding Surgery ===")
        model = embedding_surgery(model, tokenizer, clean_ref_rows,
                                  target_label=args.target_label)

    print("\n=== Evaluating poisoned weights BEFORE victim fine-tuning ===")
    pre = evaluate(model, tokenizer, clean_dev_rows, eval_sets, device)
    show("pre-victim", pre)

    print("\n=== Step 4 (victim): clean fine-tuning on the victim's own data ===")
    model = victim_finetune(model, tokenizer, victim_train_rows, device,
                             epochs=args.victim_epochs)

    print("\n=== Final evaluation AFTER victim fine-tuning ===")
    post = evaluate(model, tokenizer, clean_dev_rows, eval_sets, device)
    show("post-victim", post)

    import json
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "pre_victim": pre, "post_victim": post}, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
