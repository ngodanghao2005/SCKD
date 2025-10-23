"""
SCKD (reworked)
- Two-stage training (Supervised -> Discovery)
- Frozen replica encoder Er after stage-1
- Bidirectional distillation (L_k->n and L_n->k)
- Similarity preservation w.r.t. Er
- Sinkhorn pseudo-labeling on unlabeled
- Backbones: ResNet-18 (CIFAR/IN100) and ViT-B/16 (DINO) for fine-grained
- Optional: ViT fine-tune last block only

Notes
-----
This file focuses on model, loss, and a reference training loop skeleton.
You still need to plug in your DataLoaders (x_l, y_l) and (x_u) per batch.
"""

from __future__ import annotations
import copy, math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Tuple
from torchvision import models
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
# import timm

# ============================
# 1) Encoders
# ============================
class Encoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.out_dim = base.fc.in_features  # 512

        # elif backbone == "vit_b16_dino":
        #     # ViT-B/16 DINO from timm
        #     self.encoder = timm.create_model("vit_base_patch16_224_dino", pretrained=pretrained)
        #     self.encoder.reset_classifier(0)
        #     self.out_dim = self.encoder.num_features  # 768
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        if self.backbone_name == "resnet18":
            f = f.view(f.size(0), -1)
        return f

    # def finetune_last_block_only(self):
    #     """For ViT, freeze all but the last block (and pos_embed/cls/head if needed).
    #     For ResNet, no-op.
    #     """
    #     if self.backbone_name != "vit_b16_dino":
    #         return
    #     for name, p in self.encoder.named_parameters():
    #         # unfreeze last block (blocks.11) and norm/ln after it
    #         trainable = ("blocks.11" in name) or ("norm" in name) or ("head" in name)
    #         p.requires_grad = trainable

# ============================
# 2) Heads
# ============================
class KnownHead(nn.Module):
    def __init__(self, in_dim: int, cl: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, cl)
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f)

class NovelHead(nn.Module):
    def __init__(self, in_dim: int, cu: int, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, cu)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f)

# ============================
# 3) SCKD Model
# ============================
class SCKDModel(nn.Module):
    def __init__(self, encoder: Encoder, cl: int, cu: int, tau: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.hl = KnownHead(encoder.out_dim, cl)
        self.hu = NovelHead(encoder.out_dim, cu)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.encoder(x)                       # (B, 512): feature vector từ ResNet18
        l = self.hl(f)                            # (B, C_l): logits cho known classes
        u = self.hu(f)                            # (B, C_u): logits cho novel classes
        return f, l, u

# ============================
# 4) Utils
# ============================
@torch.no_grad()
def l2normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

@torch.no_grad()
def cosine_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return l2normalize(A) @ l2normalize(B).t()

@torch.no_grad()
def normalize_absmax(S: torch.Tensor) -> torch.Tensor:
    return S / (S.abs().max() + 1e-12)

# @torch.no_grad()
# def sinkhorn(logits: torch.Tensor, n_iter: int = 3, eps: float = 0.05) -> torch.Tensor:
#     Q = torch.exp(logits / eps).t()  # (C, B)
#     B, C = Q.shape[1], Q.shape[0]
#     Q /= Q.sum()
#     r = torch.ones(C, device=Q.device) / C
#     c = torch.ones(B, device=Q.device) / B
#     for _ in range(n_iter):
#         Q *= (r / Q.sum(1)).unsqueeze(1)
#         Q *= (c / Q.sum(0)).unsqueeze(0)
#     return (Q / Q.sum(0, keepdim=True)).t()  # (B, C)
@torch.no_grad()
def sinkhorn(logits: torch.Tensor, n_iter: int = 5, eps: float = 0.35) -> torch.Tensor:
    # ổn định số học trước khi exp
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    Q = torch.exp(logits / eps).t()  # (C, B)
    B, C = Q.shape[1], Q.shape[0]

    r = torch.ones(C, device=Q.device) / C
    c = torch.ones(B, device=Q.device) / B

    for _ in range(n_iter):
        Q /= Q.sum(1, keepdim=True) + 1e-12
        Q *= r.unsqueeze(1)
        Q /= Q.sum(0, keepdim=True) + 1e-12
        Q *= c.unsqueeze(0)

    return (Q / (Q.sum(0, keepdim=True) + 1e-12)).t()  # (B, C)

# ============================
# 5) Loss: Bidirectional Distillation + CE + Similarity Pres.
# ============================
class SCKDLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 0.5, distill_temp: float = 1.0,
                 sinkhorn_iter: int = 5, sinkhorn_eps: float = 0.35):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.distill_temp = distill_temp
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_eps = sinkhorn_eps

    def forward(self, model: SCKDModel, Er: Encoder,
                x_l: torch.Tensor, y_l: torch.Tensor,
                x_u: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Forward
        f_l, lhl, lhu = model(x_l)  # labeled pass
        f_u, uhl, uhu = model(x_u)  # unlabeled pass

        l_full = torch.cat([lhl, lhu], dim=1) # (N, Ck + Cn)
        u_full = torch.cat([uhl, uhu], dim=1) # (M, Ck + Cn)
        log_p_full = F.log_softmax(torch.cat([l_full, u_full], dim=0), dim=1) # (N+M, Ck + Cn)
        
        N = x_l.size(0)  # số labeled samples
        M = x_u.size(0)  # số unlabeled samples
        Ck = lhl.size(1)
        Cn = uhu.size(1)

        # 2) Replica features (frozen Er) - CHỈ cho labeled samples
        with torch.no_grad():
            vl = Er(x_l)  # replica features for labeled samples only
            
        # print("vl stats:", vl.mean().item(), vl.std().item())
        # print("f_u stats:", f_u.mean().item(), f_u.std().item())

        # 3) Cross-group similarity between (vl) and (current f_u) to bridge spaces
        S = normalize_absmax(cosine_matrix(vl, f_u))  # (Bl, Bu)
        
        # print("Similarity matrix - Min:", S.min().item(), "Max:", S.max().item(), "Mean:", S.mean().item())

        # 4) k->n distillation: sử dụng logits thay vì one-hot
        with torch.no_grad():
            lhu_logits = lhu.detach()

        lhat_u = self.alpha * (S.t() @ lhu_logits)  # (Bu, Cn)
        lhat_u = F.softmax(lhat_u / self.distill_temp, dim=1)
        
        # KL( known-head on x_u || soft targets from labeled )
        loss_k2n = F.kl_div(
            F.log_softmax(uhu / self.distill_temp, dim=1),
            lhat_u,
            reduction='batchmean'
        )

        # 5) n->k distillation: sử dụng logits
        with torch.no_grad():
            uhl_logits = uhl.detach()  # (Bu, Ck) - logits từ known head trên unlabeled
            
        lhat_l = self.alpha * (S @ uhl_logits)  # (Bl, Ck)
        lhat_l = F.softmax(lhat_l / self.distill_temp, dim=1)
        
        loss_n2k = F.kl_div(
            F.log_softmax(lhl / self.distill_temp, dim=1),
            lhat_l,
            reduction='batchmean'
        )

        # print("uhu output range:", uhu.min().item(), uhu.max().item())
        # # print("Sinkhorn output sum per sample:", pseudo.sum(dim=1))
        # print("uhu output sample:", uhu[0])  # Xem distribution của sample đầu tiên
        
        # Nhãn mục tiêu cho Known Samples: One-Hot (Ck) + Zeros (Cn)
        y_l_onehot = F.one_hot(y_l, num_classes=Ck) # (N, Ck)
        y_l_full = torch.cat([y_l_onehot, torch.zeros(N, Cn, device=y_l.device)], dim=1)  # (N, Ck + Cn)
        # loss_sup = F.cross_entropy(lhl, y_l_full.argmax(dim=1))  # supervised CE loss on known head

        # 6) Pseudo-label CE on unlabeled via Sinkhorn from novel-head
        with torch.no_grad():
            pseudo = sinkhorn(uhu.detach(), n_iter=self.sinkhorn_iter, eps=self.sinkhorn_eps)
        #     y_u_hat = pseudo.argmax(dim=1)
        # loss_unl = F.cross_entropy(uhu, y_u_hat)
        
        all_preds = []
        print("Sinkhorn output range:", pseudo.min().item(), pseudo.max().item())
        # print("Sinkhorn output sum per sample:", pseudo.sum(dim=1))
        print("Sinkhorn output sample:", pseudo[0])  # Xem distribution của sample đầu tiên
        all_preds.extend(pseudo.argmax(dim=1).cpu().tolist())
        all_preds = np.array(all_preds)
        print("counts preds:", {k:int((all_preds==k).sum()) for k in np.unique(all_preds)})
        print()
        
        # entropy = - (pseudo * torch.log(pseudo + 1e-12)).sum(dim=1).mean()
        
        # loss_unl = F.kl_div(
        #     F.log_softmax(uhu, dim=1),
        #     pseudo,
        #     reduction='batchmean'
        # )
        
        y_u_full = torch.cat([torch.zeros(M, Ck, dtype=pseudo.dtype, device=pseudo.device), pseudo], dim=1) # (M, Ck + Cn)
        y_full = torch.cat([y_l_full, y_u_full], dim=0) # (N+M, Ck + Cn)
        
        # log_probs = F.log_softmax(uhu, dim=1)       # Tính log probability của model
        # loss_unl = (- (pseudo * log_probs).sum(dim=1)).mean()  # CE với pseudo-labels từ Sinkhorn
        
        # loss_unl = F.kl_div(
        #     F.log_softmax(uhu, dim=1),
        #     pseudo,
        #     reduction='batchmean'
        # )
        # ) - 0.1 * entropy
        
        # print(f"Pseudo-label entropy: {entropy.item():.4f}")
        # print()
            # loss_unl = torch.mean(torch.sum(-pseudo * F.log_softmax(uhu, dim=1), dim=1))

        # Aggregate
        loss_ce_total = -torch.sum(y_full * log_p_full) / (N + M)
        loss_total = loss_ce_total + self.beta * (loss_k2n + loss_n2k)

        stats = {
            "loss_total": float(loss_total.item()),
            "loss_ce_total": float(loss_ce_total.item()),
            "loss_k2n": float(loss_k2n.item()),
            "loss_n2k": float(loss_n2k.item()),
        }
        return loss_total, stats

# ============================
# 6) Optimizer & Scheduler
# ============================
def get_optimizer_scheduler(model: SCKDModel, mode: str, cfg: Dict, total_epochs: int):
    if mode == "cifar":
        base_lr = cfg.get('lr', 0.4)          # đỉnh: 0.4
        min_lr  = cfg.get('min_lr', 1e-3)     # đáy: 0.001
        warmup  = cfg.get('warmup_epochs', 10)
        momentum, weight_decay = 0.9, 1.5e-4  # theo paper

        opt = SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        # start_factor = min_lr / base_lr       # 0.001 / 0.4 = 0.0025
        step = (base_lr - min_lr) / (warmup - 1)
        def lr_lambda(epoch):
            if epoch < warmup:
                # warm-up tuyến tính: 0.001 -> 0.4 trong 10 epoch
                # return start_factor + (1.0 - start_factor) * (epoch + 1) / warmup
                return (min_lr + step * epoch) / base_lr
            # cosine: 0.4 -> 0.001 đến hết total_epochs
            t = (epoch - warmup) / max(1, total_epochs - warmup)
            return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * 0.5 * (1 + math.cos(math.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # else:
    #     # Fine-grained (ViT-B/16 + AdamW): 1e-4 -> 1e-3 (10 ep) rồi cosine về 1e-4 @ 100 ep
    #     base_lr = cfg.get('lr', 1e-3)
    #     min_lr  = cfg.get('min_lr', 1e-4)
    #     warmup  = cfg.get('warmup_epochs', 10)
    #     opt = AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.get('wd', 0.05))
    #     start_factor = min_lr / base_lr
    #     def lr_lambda(epoch):
    #         if epoch < warmup:
    #             return start_factor + (1.0 - start_factor) * (epoch + 1) / warmup
    #         t = (epoch - warmup) / max(1, total_epochs - warmup)
    #         return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * 0.5 * (1 + math.cos(math.pi * t))
    #     sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched

# ============================
# 7) Build model
# ============================
def build_model(mode: str, cl: int, cu: int, device: str = "cuda", vit_last_block_only: bool = True) -> SCKDModel:
    if mode == "cifar":
        enc = Encoder("resnet18", pretrained=True)
    # else:
    #     enc = Encoder("vit_b16_dino", pretrained=True)
    #     if vit_last_block_only:
    #         enc.finetune_last_block_only()
    model = SCKDModel(enc, cl, cu, tau=0.1).to(device)
    return model

# ============================
# 8) Stage-1 & Stage-2 training skeletons
# ============================
@torch.no_grad()
def clone_frozen_replica(encoder: Encoder) -> Encoder:
    Er = copy.deepcopy(encoder)
    for p in Er.parameters():
        p.requires_grad = False
    return Er


def stage1_supervised_epoch(model: SCKDModel, loader_labeled, optimizer, device: str = "cuda") -> Dict[str, float]:
    model.train()
    total, correct, loss_accum, n = 0, 0, 0.0, 0
    for x, y in loader_labeled:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, lhl, _ = model(x)
        loss = F.cross_entropy(lhl, y)
        loss.backward()
        optimizer.step()
        loss_accum += float(loss.item()) * x.size(0)
        pred = lhl.argmax(1)
        correct += int((pred == y).sum().item())
        n += x.size(0)
    return {"sup_loss": loss_accum / max(1, n), "sup_acc": correct / max(1, n)}


def stage2_discovery_epoch(model: SCKDModel, Er: Encoder, loader_labeled, loader_unlabeled,
                           optimizer, criterion: SCKDLoss, device: str = "cuda") -> Dict[str, float]:
    model.train()
    it_u = iter(loader_unlabeled)
    stats_accum: Dict[str, float] = {}
    count = 0
    for x_l, y_l in loader_labeled:
        try:
            x_u = next(it_u)
        except StopIteration:
            it_u = iter(loader_unlabeled)
            x_u = next(it_u)
        if isinstance(x_u, (list, tuple)):
            x_u = x_u[0]
        x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
        
        if count == 0:  # Chỉ in batch đầu tiên
            print("Batch - Labeled classes:", torch.unique(y_l))
            print("Batch - Labeled class distribution:", torch.bincount(y_l))

        optimizer.zero_grad(set_to_none=True)
        loss, stats = criterion(model, Er, x_l, y_l, x_u)
        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            stats_accum[k] = stats_accum.get(k, 0.0) + v
        count += 1

    for k in list(stats_accum.keys()):
        stats_accum[k] /= max(1, count)
    return stats_accum

@torch.no_grad()
def evaluate_task_aware(model: SCKDModel, loader_unlabeled, num_novel_classes: int, device: str = "cuda",
                        sinkhorn_iter: int = 5, sinkhorn_eps: float = 0.35) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []

    for x_u, y_u in loader_unlabeled:
        x_u, y_u = x_u.to(device), y_u.to(device)
        _, _, uhu = model(x_u)   # lấy logits từ Novel-Class Head
        pseudo = sinkhorn(uhu.detach(), n_iter=sinkhorn_iter, eps=sinkhorn_eps)
        pred = pseudo.argmax(1)
        # print(pred)
        # print(y_u)
        # print()

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(y_u.cpu().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    min_label = min(all_targets)
    all_targets_mapped = all_targets - min_label
    
    print("unique targets:", np.unique(all_targets))
    print("unique preds:", np.unique(all_preds))
    print("counts targets:", {k:int((all_targets==k).sum()) for k in np.unique(all_targets)})
    print("counts preds:", {k:int((all_preds==k).sum()) for k in np.unique(all_preds)})

    # === Hungarian matching ===
    # --- REMAP ground-truth novel labels to 0..Cu-1 ---
    # uniq_targets = np.unique(all_targets)
    # # if they are contiguous like [5,6,7,8,9], this will map to 0..4 by subtracting min
    # if np.array_equal(uniq_targets, np.arange(uniq_targets.min(), uniq_targets.min() + uniq_targets.size)):
    #     all_targets_mapped = all_targets - uniq_targets.min()
    # else:
    #     # general mapping
    #     mapping = {int(v): i for i, v in enumerate(uniq_targets)}
    #     all_targets_mapped = np.array([mapping[int(v)] for v in all_targets])

    # Cu = len(np.unique(all_preds))  # or set Cu manually
    # cm = confusion_matrix(all_targets_mapped, all_preds, labels=np.arange(Cu))
    cm = confusion_matrix(all_targets_mapped, all_preds, labels=np.arange(num_novel_classes))
    # n_true, n_pred = cm.shape
    # if n_pred != n_true:
    #     # Pad ma trận để thành vuông
    #     size = max(n_pred, n_true)
    #     padded = np.zeros((size, size), dtype=np.int64)
    #     padded[:n_true, :n_pred] = cm
    #     cm = padded
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()
    return {"acc_unlabeled": acc}

# ============================
# 9) Example high-level train() wrapper (pseudo)
# ============================
"""
Example usage (you must provide DataLoaders):

model = build_model(mode, cl, cu, device)
opt1, sch1 = get_optimizer_scheduler(model, mode, cfg_stage1, total_epochs=E1)
for e in range(E1):
    m = stage1_supervised_epoch(model, loader_labeled, opt1, device)
    sch1.step()

Er = clone_frozen_replica(model.encoder)  # freeze replica after stage-1

criterion = SCKDLoss(alpha=0.1, beta=0.5, distill_temp=1.0,
                     sinkhorn_iter=3, sinkhorn_eps=0.05)
opt2, sch2 = get_optimizer_scheduler(model, mode, cfg_stage2, total_epochs=E2)
for e in range(E2):
    stats = stage2_discovery_epoch(model, Er, loader_labeled, loader_unlabeled, opt2, criterion, device)
    sch2.step()
"""
