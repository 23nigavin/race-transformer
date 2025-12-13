import torch 
import os
import math
import time
import torch.nn.functional as F
from tqdm import tqdm
torch.set_float32_matmul_precision('high')

from .vit_config import VISION_CONFIG
from .vit_data import get_data_food101
from .vit_models import VisionTransformer
    
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR for `warmup_steps` optimizer updates,
    then linear decay to 0 by `total_steps`. Call scheduler.step() *after* optimizer.step().
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # count optimizer steps
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
    grad_accum_steps: int = 1
):
    """
    Classification-friendly training loop with:
      - gradient accumulation
      - linear warmup + linear decay LR schedule (per optimizer step)
    """
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []
    train_times, val_times = [], []

    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    out_path = f"trial_K{K}_L{L}_M{M}_VIT.txt"

    steps_per_epoch = len(train_loader)                          # micro-steps
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)  # optimizer steps
    total_updates  = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.1 * total_updates))           # 10% warmup

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    def _log(fp, msg):
        print(msg); fp.write(msg + "\n"); fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Epochs: {num_epochs}")
        _log(f, "-" * 72)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # === TRAIN ===
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum_count = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)                  # [B, C]
                loss = F.cross_entropy(outputs, labels)  # classification CE

                # scale for accumulation
                (loss / grad_accum_steps).backward()
                accum_count += 1

                # metrics (unscaled)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                # update if we've accumulated enough micro-steps
                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()  # step LR *per optimizer step*
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            # flush any remainder
            if accum_count > 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0
            train_times.append(train_time)

            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)
            train_losses.append(tr_l)
            train_accs.append(tr_a)

            # === VALIDATION ===
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1
            val_times.append(val_time)

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            val_losses.append(va_l)
            val_accs.append(va_a)

            # current lr (take the first group)
            curr_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]

            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                 f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 72)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")

    return {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc":  train_accs,   "val_acc":  val_accs,
        "train_time": train_times,  "val_time": val_times,
    }


def start_experiment():
    device = "cuda:1"
    train_loader, val_loader, info = get_data_food101(batch_size=VISION_CONFIG["batch_size"])
    num_epochs = 100

    # print("Training Softmax model...")
    # torch.manual_seed(123)
    # model_gpt = VisionTransformer(VISION_CONFIG, "softmax")
    # model_gpt.to(device)
    # optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_gpt = train_model_simple(
    #     model_gpt, train_loader, val_loader, optimizer_gpt, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=4
    # )

    # print("Training RACE model...")
    # torch.manual_seed(123)
    # model_race = VisionTransformer(VISION_CONFIG, "race")
    # model_race.to(device)
    # optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_race = train_model_simple(
    #    model_race, train_loader, val_loader, optimizer_race, device,
    #    num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=4
    # )

    print("Training MACH model...")
    torch.manual_seed(123)
    model_race = VisionTransformer(VISION_CONFIG, "mach")
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=3e-4, weight_decay=0.001)

    metrics_race = train_model_simple(
       model_race, train_loader, val_loader, optimizer_race, device,
       num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    )

    # print("Training Linformer model...")
    # torch.manual_seed(123)
    # model_linformer = VisionTransformer(VISION_CONFIG, "linformer")
    # model_linformer.to(device)
    # optimizer_race = torch.optim.AdamW(model_linformer.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_race = train_model_simple(
    #    model_linformer, train_loader, val_loader, optimizer_race, device,
    #    num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

    # print("Training LinearAttention...")
    # torch.manual_seed(123)
    # model_linear = VisionTransformer(VISION_CONFIG, "linear")
    # print(sum(p.numel() for p in model_linear.parameters() if p.requires_grad))
    # model_linear.to(device)
    # optimizer_linear = torch.optim.AdamW(model_linear.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_linear = train_model_simple(
    #    model_linear, train_loader, val_loader, optimizer_linear, device,
    #    num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

    # print("Training Angular Attention....")
    # torch.manual_seed(123)
    # model_angular = torch.compile(VisionTransformer(VISION_CONFIG, "angular"))
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_angular = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=4
    # )

    # print("Training Performer Attention....")
    # torch.manual_seed(123)
    # model_performer = VisionTransformer(VISION_CONFIG, "performer")
    # model_performer.to(device)
    # optimizer_performer = torch.optim.AdamW(model_performer.parameters(), lr=3e-4, weight_decay=0.001)

    # metrics_performer = train_model_simple(
    #     model_performer, train_loader, val_loader, optimizer_performer, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

if __name__ == "__main__":
    start_experiment()
