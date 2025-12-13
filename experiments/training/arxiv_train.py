# ==================================================
# 0) Imports & Global Config
# ==================================================
import math, time, os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .arxiv_config import TEXT_CONFIG
from .arxiv_models import TextTransformerClassifier
from .arxiv_config import DEVICE
from .arxiv_data import train_dl, test_dl, PACK_MIN_FRAC, max_len
from .arxiv_data import compute_effective_lengths_from_loader, print_effective_length_stats


# ==================================================
# 8) Scheduler & training loop (like vision file)
# ==================================================
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    cfg,
    attn_type,
    grad_accum_steps: int = 1,
):
    steps_per_epoch   = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates     = num_epochs * updates_per_epoch
    warmup_updates    = max(1, int(0.1 * total_updates))

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    out_path = f"arxiv_{attn_type}_644K.txt"

    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Attn: {attn_type}, Epochs: {num_epochs}")
        _log(f, "-" * 80)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # ---- TRAIN ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum_count = 0

            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch} [train]",
                leave=False,
            )

            for tokens, masks, labels in train_iter:
                tokens  = tokens.to(device)
                masks   = masks.to(device)
                labels  = labels.to(device)

                logits = model(tokens, masks)
                loss   = F.cross_entropy(logits, labels)

                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                # Update tqdm with running stats
                train_iter.set_postfix({
                    "loss": running_loss / max(1, len(train_iter)),
                    "acc":  running_correct / max(1, running_total),
                })

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            if accum_count > 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0

            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)

            # ---- VAL ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total   = 0

            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]",
                leave=False,
            )

            with torch.no_grad():
                for tokens, masks, labels in val_iter:
                    tokens = tokens.to(device)
                    masks  = masks.to(device)
                    labels = labels.to(device)

                    logits = model(tokens, masks)
                    loss   = F.cross_entropy(logits, labels)
                    val_loss_total += loss.item()

                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

                    val_iter.set_postfix({
                        "loss": val_loss_total / max(1, len(val_iter)),
                        "acc":  val_correct / max(1, val_total),
                    })

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            curr_lr = scheduler.get_last_lr()[0]

            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                 f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 80)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")


# ==================================================
# 9) Run all baselines (like vision)
# ==================================================
def run_experiment(attn_types, cfg):
    for attn_type in attn_types:
        print(f"\n=== Training {attn_type.upper()} on Arxiv 64K ===")
        model = TextTransformerClassifier(cfg, attn_type).to(DEVICE)
        opt   = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        train_model_simple(
            model=model,
            train_loader=train_dl,
            val_loader=test_dl,
            optimizer=opt,
            device=DEVICE,
            num_epochs=cfg["epochs"],
            cfg=cfg,
            attn_type=attn_type,
            grad_accum_steps=cfg["grad_accum_steps"],
        )

if __name__ == "__main__":
    # same baseline set as Vision file
    run_experiment(
        ["mach"],
        TEXT_CONFIG,
    )
