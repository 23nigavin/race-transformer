# ==================================================
# 0) Imports & Global Config
# ==================================================
import re, random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from collections import Counter, defaultdict
from typing import List
import numpy as np

from .arxiv_config import (
    SEED, DEVICE, DATASET_NAME, TEXT_FIELD, LABEL_FIELD,
    PACK_TARGET_LEN, PACK_MIN_FRAC, MIN_DOC_LEN,
    DESIRED_TRAIN_TOTAL, DESIRED_TEST_TOTAL, TEXT_CONFIG,
)

# ==================================================
# 1) Tokenizer (basic_english)
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])   # punctuation
     |(\d+[%]?)                                  # numbers (and percent)
     |([A-Za-z]+(?:'[A-Za-z]+)?)                 # words w/ optional apos
    """,
    re.VERBOSE,
)

def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    for punc, num, word in _basic_english_re.findall(text):
        if punc:
            tokens.append(punc)
        elif num:
            tokens.append(num)
        elif word:
            tokens.append(word)
    return tokens

tok = basic_english_tokenizer

# ==================================================
# 2) (Optional) EDA augmenters
# ==================================================
def eda_random_deletion(tokens, p=0.05):
    if len(tokens) == 1:
        return tokens
    out = [t for t in tokens if random.random() > p]
    return out or [random.choice(tokens)]

def eda_random_swap(tokens, n_swaps=3):
    toks = tokens.copy()
    for _ in range(n_swaps):
        if len(toks) < 2:
            break
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks

# ==================================================
# 3) Helper: length stats
# ==================================================
def print_length_stats(arr: np.ndarray, name: str, thresholds=()):
    print("--------------------------------------------------")
    print(f"Raw token length stats ({name})")
    print("--------------------------------------------------")
    print(f"Count:      {len(arr)}")
    print(f"Min:        {int(arr.min())}")
    print(f"Max:        {int(arr.max())}")
    print(f"Mean:       {float(arr.mean()):.1f}")
    print(f"Median:     {int(np.median(arr))}")
    print(f"90th pct:   {int(np.percentile(arr, 90))}")
    print(f"95th pct:   {int(np.percentile(arr, 95))}")
    print(f"99th pct:   {int(np.percentile(arr, 99))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

# ==================================================
# 4) Balanced subset with min-length constraint (per doc)
#    (IDENTICAL behavior to your "good" script)
# ==================================================
def make_balanced_long_examples(split, desired_total, min_len, name="train", seed=SEED):
    """
    Make a class-balanced subset where each *doc* has raw token length >= min_len.
    Returns:
      examples: list[(label, text)]
      num_classes: int
    """
    labels = list(split[LABEL_FIELD])
    texts  = list(split[TEXT_FIELD])

    print(f"\nBuilding balanced LONG-{name} subset with min_len = {min_len}...")
    print(f"Original {name} split size: {len(labels)}")

    # Precompute lengths
    print(f"Tokenizing {name} split to compute lengths...")
    lengths = []
    for txt in texts:
        toks = tok(str(txt))
        lengths.append(len(toks))
    lengths = np.array(lengths, dtype=np.int32)

    # Bucket by class, keeping only long docs
    buckets = defaultdict(list)
    for idx, (y, L) in enumerate(zip(labels, lengths)):
        y_int = int(y)
        if L >= min_len:
            buckets[y_int].append(idx)

    num_classes = len(buckets)
    if num_classes == 0:
        raise ValueError(f"No examples meet min_len = {min_len} in {name} split!")

    print(f"Found {num_classes} classes with at least one example >= min_len.")
    for y in sorted(buckets.keys()):
        print(f"  Class {y}: {len(buckets[y])} examples >= {min_len}")

    # Compute per-class quota
    max_possible_per_class = min(len(idxs) for idxs in buckets.values())
    desired_per_class      = desired_total // num_classes
    per_class              = min(max_possible_per_class, desired_per_class)

    if per_class == 0:
        raise ValueError(
            f"min_len = {min_len} is too strict: at least one class has 0 long examples."
        )

    actual_total = per_class * num_classes
    print(f"\nDesired total {name} examples: {desired_total}")
    print(f"Max possible per class (given min_len): {max_possible_per_class}")
    print(f"Using per_class = {per_class}, so actual total = {actual_total}")

    # Sample per class
    rng = random.Random(seed)
    chosen_idx = []
    for y, idxs in buckets.items():
        rng.shuffle(idxs)
        chosen_idx.extend(idxs[:per_class])
    rng.shuffle(chosen_idx)

    examples = [(int(labels[i]), texts[i]) for i in chosen_idx]

    # Stats for the final subset of docs
    final_lengths = lengths[chosen_idx]
    print_length_stats(
        final_lengths,
        f"{name} docs (balanced, length-filtered)",
        thresholds=(min_len,),
    )

    return examples, num_classes

# ==================================================
# 5) Streaming packer: use all tokens up to 64k chunks
#    (IDENTICAL behavior to your "good" script)
# ==================================================
def pack_examples_streaming(
    examples,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED,
):
    """
    Streaming packer that:
      - groups docs by label
      - iterates through docs per class, tokenizing and appending into a buffer
      - emits a packed example every time buffer hits target_len
      - emits a final partial example if it's >= min_frac * target_len

    This reuses residual tokens from long docs rather than discarding them.
    """
    rng = random.Random(seed)
    per_class_docs = defaultdict(list)

    for lbl, txt in examples:
        per_class_docs[int(lbl)].append(str(txt))

    new_examples = []

    for lbl, docs in per_class_docs.items():
        # Shuffle docs within class to randomize packing
        rng.shuffle(docs)

        cur_tokens = []
        for txt in docs:
            toks = tok(txt)
            j = 0
            n = len(toks)
            while j < n:
                remaining_space = target_len - len(cur_tokens)
                if remaining_space <= 0:
                    # Buffer full → emit
                    if len(cur_tokens) >= int(min_frac * target_len):
                        new_examples.append((lbl, " ".join(cur_tokens)))
                    cur_tokens = []
                    remaining_space = target_len

                take = min(remaining_space, n - j)
                if take <= 0:
                    break

                cur_tokens.extend(toks[j : j + take])
                j += take

                if len(cur_tokens) == target_len:
                    # Emit full packed sequence
                    new_examples.append((lbl, " ".join(cur_tokens)))
                    cur_tokens = []

        # End of docs for this class: flush leftover if big enough
        if len(cur_tokens) >= int(min_frac * target_len):
            new_examples.append((lbl, " ".join(cur_tokens)))
        # else: drop tiny tail

    return new_examples

# ==================================================
# 6) Dataset class (same as "good" script)
# ==================================================
class ArxivDataset(Dataset):
    def __init__(self, examples, max_len, stoi, pad_idx=0, unk_idx=1, augment=False):
        self.examples = examples
        self.max_len  = max_len
        self.augment  = augment
        self.stoi     = stoi
        self.pad_idx  = pad_idx
        self.unk_idx  = unk_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = tok(str(txt))

        if self.augment:
            op = random.choice(["del", "swap", None])
            if op == "del":
                toks = eda_random_deletion(toks)
            elif op == "swap":
                toks = eda_random_swap(toks)

        toks = toks[: self.max_len]
        ids  = [self.stoi.get(t, self.unk_idx) for t in toks]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return int(lbl), torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        masks  = (tokens != self.pad_idx).long()
        return tokens, masks, torch.tensor(labels, dtype=torch.long)

# ==================================================
# 7) Effective length stats from DataLoader masks
# ==================================================
def compute_effective_lengths_from_loader(dl, num_batches=100):
    all_lengths = []
    for i, (tokens, masks, labels) in enumerate(dl):
        lens = masks.sum(dim=1).cpu().numpy()
        all_lengths.append(lens)
        if i + 1 >= num_batches:
            break
    if not all_lengths:
        return np.array([], dtype=np.int32)
    return np.concatenate(all_lengths).astype(np.int32)

def print_effective_length_stats(arr: np.ndarray, max_len: int, thresholds=()):
    print("--------------------------------------------------")
    print("Effective sequence length stats (after padding/truncation)")
    print("--------------------------------------------------")
    print(f"Count (sampled): {len(arr)}")
    if len(arr) == 0:
        print("No data collected from DataLoader.")
        return
    print(f"Min:             {int(arr.min())}")
    print(f"Max:             {int(arr.max())}  (max_len = {max_len})")
    print(f"Mean:            {float(arr.mean()):.1f}")
    print(f"Median:          {int(np.median(arr))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

print("Loading dataset:", DATASET_NAME)
raw = load_dataset(DATASET_NAME)

if "validation" in raw:
    train_split = raw["train"]
    test_split  = raw["validation"]
elif "test" in raw:
    train_split = raw["train"]
    test_split  = raw["test"]
else:
    tmp = raw["train"].train_test_split(test_size=0.2, seed=SEED)
    train_split, test_split = tmp["train"], tmp["test"]

# -----------------------------------------------
# 8.1) Build balanced, long-doc subsets (>= MIN_DOC_LEN)
# -----------------------------------------------
train_docs, num_classes_train = make_balanced_long_examples(
    train_split,
    desired_total=DESIRED_TRAIN_TOTAL,
    min_len=MIN_DOC_LEN,
    name="train",
    seed=SEED,
)
test_docs, num_classes_test = make_balanced_long_examples(
    test_split,
    desired_total=DESIRED_TEST_TOTAL,
    min_len=MIN_DOC_LEN,
    name="test",
    seed=SEED,
)

assert num_classes_train == num_classes_test
num_classes = num_classes_train

print(f"Final balanced-long train docs (>= {MIN_DOC_LEN}): {len(train_docs)}")
print(f"Final balanced-long test docs  (>= {MIN_DOC_LEN}): {len(test_docs)}")
print(f"Num classes: {num_classes}\n")

# -----------------------------------------------
# 8.2) STREAMING: pack docs into ~64k sequences per class
# -----------------------------------------------
print(f"Streaming-pack long docs into ~{PACK_TARGET_LEN} token sequences...")
train_examples_packed = pack_examples_streaming(
    train_docs,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED,
)
test_examples_packed = pack_examples_streaming(
    test_docs,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED + 1,
)

print(f"Packed train size (~{PACK_TARGET_LEN}): {len(train_examples_packed)}")
print(f"Packed test size  (~{PACK_TARGET_LEN}): {len(test_examples_packed)}\n")

# Stats on packed sequences (raw token counts)
def packed_lengths(examples):
    return np.array([len(tok(str(txt))) for _, txt in examples], dtype=np.int32)

train_packed_lengths = packed_lengths(train_examples_packed)
test_packed_lengths  = packed_lengths(test_examples_packed)

print_length_stats(
    train_packed_lengths,
    name="train_packed (~64k)",
    thresholds=(int(PACK_MIN_FRAC * PACK_TARGET_LEN), PACK_TARGET_LEN),
)
print_length_stats(
    test_packed_lengths,
    name="test_packed (~64k)",
    thresholds=(int(PACK_MIN_FRAC * PACK_TARGET_LEN), PACK_TARGET_LEN),
)

# Use packed examples from here on
train_examples = train_examples_packed
test_examples  = test_examples_packed

# -----------------------------------------------
# 8.3) Build vocab from packed train examples
# -----------------------------------------------
print("Building vocabulary from packed train examples...")
counter = Counter()
for lbl, txt in train_examples:
    counter.update(tok(str(txt)))

most_common = [w for w, _ in counter.most_common(TEXT_CONFIG["vocab_limit"])]
stoi = {w: i + 2 for i, w in enumerate(most_common)}
stoi["<pad>"] = 0
stoi["<unk>"] = 1

PAD_IDX, UNK_IDX = 0, 1
VOCAB_SIZE = len(stoi)
TEXT_CONFIG["vocab_size"]  = VOCAB_SIZE
TEXT_CONFIG["num_classes"] = num_classes

print(f"Vocab size: {VOCAB_SIZE}\n")

# -----------------------------------------------
# 8.4) Create datasets / loaders at 64k
# -----------------------------------------------
max_len  = TEXT_CONFIG["max_len"]  # 64_000
batch_sz = 2  # super long sequences → small batch

train_ds = ArxivDataset(
    train_examples,
    max_len=max_len,
    stoi=stoi,
    pad_idx=PAD_IDX,
    unk_idx=UNK_IDX,
    augment=True,
)
test_ds = ArxivDataset(
    test_examples,
    max_len=max_len,
    stoi=stoi,
    pad_idx=PAD_IDX,
    unk_idx=UNK_IDX,
    augment=False,
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_sz,
    shuffle=True,
    drop_last=True,
    pin_memory=(DEVICE == "cuda"),
    num_workers=4,
    collate_fn=train_ds.collate_fn,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_sz,
    shuffle=False,
    pin_memory=(DEVICE == "cuda"),
    num_workers=2,
    collate_fn=test_ds.collate_fn,
)

print(f"Train batches: {len(train_dl)}")
print(f"Test  batches: {len(test_dl)}\n")

# -----------------------------------------------
# 8.5) Effective length stats from DataLoader
# -----------------------------------------------
if __name__ == "__main__":
    eff_lengths = compute_effective_lengths_from_loader(
        train_dl,
        num_batches=100,  # sample
    )
    print_effective_length_stats(
        eff_lengths,
        max_len=max_len,
        thresholds=(int(PACK_MIN_FRAC * max_len), max_len),
    )
    print("Done. Packed sequences are ~62k tokens long with minimal padding.")
