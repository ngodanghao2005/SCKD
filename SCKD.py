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
from sklearn.cluster import KMeans

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.encoder(x)                       # (B, 512): feature vector từ ResNet18
        l = self.hl(f)                            # (B, C_l): logits cho known classes
        u = self.hu(f)                            # (B, C_u): logits cho novel classes
        combined = torch.cat([l, u], dim=1)
        combined_norm = F.normalize(combined, p=2, dim=1) * 10.0  # chuẩn hóa và scale
        l = combined_norm[:, :self.hl.fc.out_features] 
        u = combined_norm[:, self.hl.fc.out_features:]
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

@torch.no_grad()
def sinkhorn(logits: torch.Tensor, n_iter: int = 5, eps: float = 0.05) -> torch.Tensor:
    # ổn định số học trước khi exp
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    Q = torch.exp(logits / eps).t()  # (C, B)
    B, C = Q.shape[1], Q.shape[0]

    c = torch.ones(C, device=Q.device) / C
    r = torch.ones(B, device=Q.device) / B

    for _ in range(n_iter):
        Q /= Q.sum(1, keepdim=True) + 1e-12
        Q *= c.unsqueeze(1)
        Q /= Q.sum(0, keepdim=True) + 1e-12
        Q *= r.unsqueeze(0)

    return (Q / (Q.sum(0, keepdim=True) + 1e-12)).t()  # (B, C)

def cdc_loss(uhu, lhu, S, alpha, T=1.0):
    # 1. Tính xác suất từ Novel-head cho các mẫu Unlabeled (Dự đoán thực tế)
    p_u = F.softmax(uhu / T, dim=1)  # [Bu, Cn]
    
    # 2. Tổng hợp "Tri thức từ Known" để tạo mục tiêu cho Novel (Giả nhãn mềm)
    with torch.no_grad():
        lhat_u_logits = alpha * torch.matmul(S.t(), lhu.detach())
        q_u = F.softmax(lhat_u_logits / T, dim=1)  # [Bu, Cn]

    # 3. Xây dựng ma trận tương đồng cặp (Pairwise Similarity) theo PCR
    # Ma trận U: Mối quan hệ giữa các mẫu do Novel-head tự cảm nhận
    # Ma trận V: Mối quan hệ giữa các mẫu được dẫn dắt bởi Known-space
    U = torch.matmul(p_u, p_u.t())  # [Bu, Bu]
    V = torch.matmul(q_u, q_u.t())  # [Bu, Bu]

    # 4. Tính KL Divergence trên ma trận tương đồng (CDC Loss)
    loss_cdc = F.kl_div(
        F.log_softmax(U, dim=1), 
        F.softmax(V, dim=1), 
        reduction='batchmean'
    ) * (T * T)

    return loss_cdc

# ============================
# 5) Loss: Bidirectional Distillation + CE + Similarity Pres.
# ============================
class SCKDLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 0.5, gamma: float = 0.5, distill_temp: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.distill_temp = distill_temp

    def forward(self, model, Er, x_l, y_l, x_u, u_idx, global_labels) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Forward
        f_l, lhl, lhu = model(x_l)  # labeled pass
        f_u, uhl, uhu = model(x_u)  # unlabeled pass

        # l_full = torch.cat([lhl, lhu], dim=1) # (N, Ck + Cn)
        # u_full = torch.cat([uhl, uhu], dim=1) # (M, Ck + Cn)
        # full = torch.cat([l_full, u_full], dim=0) / self.distill_temp # (N+M, Ck + Cn)
        # log_p_full = F.log_softmax(full, dim=1) # (N+M, Ck + Cn)
        
        N = x_l.size(0)
        M = x_u.size(0)

        with torch.no_grad():
            vl = Er(x_l)  # replica features for labeled samples only

        # Cross-group similarity between (vl) and (current f_u) to bridge spaces
        S = normalize_absmax(cosine_matrix(vl, f_u))

        # k->n distillation: sử dụng logits thay vì one-hot
        with torch.no_grad():
            lhu_logits = lhu.detach()

        lhat_u = self.alpha * (S.t() @ lhu_logits)
        lhat_u = F.softmax(lhat_u / 1.0, dim=1)
        
        loss_k2n = F.kl_div(
            F.log_softmax(uhu / 1.0, dim=1),
            lhat_u,
            reduction='batchmean'
        )

        loss_cdc = cdc_loss(uhu, lhu, S, self.alpha, T=1.0)

        # n->k distillation: sử dụng logits
        with torch.no_grad():
            uhl_logits = uhl.detach()
            
        lhat_l = self.alpha * (S @ uhl_logits)
        lhat_l = F.softmax(lhat_l / 1.0, dim=1)
        
        loss_n2k = F.kl_div(
            F.log_softmax(lhl / 1.0, dim=1),
            lhat_l,
            reduction='batchmean'
        )
        
        # Tính toán Cross-Entropy cho cả hai tập
        # Unlabeled CE dùng nhãn từ K-means
        
        loss_ce_u = F.cross_entropy(uhu / self.distill_temp, global_labels[u_idx])
        loss_ce_l = F.cross_entropy(lhl / self.distill_temp, y_l)
        
        # N và M là batch_size của từng tập
        loss_ce_total = (loss_ce_l * N + loss_ce_u * M) / (N + M)
        loss_total = loss_ce_total + self.beta * (loss_k2n + loss_n2k) + self.gamma * loss_cdc

        stats = {
            "loss_total": float(loss_total.item()),
            "loss_ce_total": float(loss_ce_total.item()),
            "loss_k2n": float(loss_k2n.item()),
            "loss_n2k": float(loss_n2k.item()),
            "loss_cdc": float(loss_cdc.item())
        }
        return loss_total, stats

# ============================
# 6) Optimizer & Scheduler
# ============================
def get_optimizer_scheduler(model: SCKDModel, mode: str, cfg: Dict, total_epochs: int):
    if mode == "cifar":
        base_lr = cfg.get('lr')
        min_lr  = cfg.get('min_lr')
        warmup  = cfg.get('warmup_epochs', 10)
        momentum, weight_decay = 0.9, 1.5e-4  # theo paper

        opt = SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
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
    correct, loss_accum, n = 0, 0.0, 0
    for x, y, _ in loader_labeled:
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

@torch.no_grad()
def run_global_estep(model, loader, num_clusters, device):
    model.eval()
    all_features = []
    # Loader này không cần shuffle, dùng để quét toàn bộ tập Unlabeled
    for x, _, _ in loader:
        f, _, _ = model(x.to(device))
        all_features.append(f.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    all_features = F.normalize(all_features, p=2, dim=1)
    all_features = all_features.numpy()
    # Phân cụm dựa trên đặc trưng hiện tại của Encoder
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(all_features)
    return torch.from_numpy(labels).long().to(device)

def stage2_discovery_epoch(model: SCKDModel, Er: Encoder, loader_labeled, loader_unlabeled,
                           optimizer, criterion: SCKDLoss, global_labels: torch.Tensor, device: str = "cuda") -> Dict[str, float]:
    model.train()
    it_u = iter(loader_unlabeled)
    stats_accum: Dict[str, float] = {}
    count = 0
    for x_l, y_l, _ in loader_labeled:
        try:
            x_u, _, u_idx = next(it_u)
        except StopIteration:
            it_u = iter(loader_unlabeled)
            x_u, _, u_idx = next(it_u)
        # if isinstance(x_u, (list, tuple)):
        #     x_u = x_u[0]
        x_l, y_l, x_u, u_idx = x_l.to(device), y_l.to(device), x_u.to(device), u_idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, stats = criterion(model, Er, x_l, y_l, x_u, u_idx, global_labels)
        
        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            stats_accum[k] = stats_accum.get(k, 0.0) + v
        count += 1
    
    for k in list(stats_accum.keys()):
        stats_accum[k] /= max(1, count)
    return stats_accum

@torch.no_grad()
def evaluate_training_subset(model: SCKDModel, loader_unlabeled, num_novel_classes: int, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []

    for x_u, y_u, _ in loader_unlabeled:
        x_u, y_u = x_u.to(device), y_u.to(device)
        _, _, uhu = model(x_u)

        pred = uhu.argmax(1)

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(y_u.cpu().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    min_label = min(all_targets)
    all_targets_mapped = all_targets - min_label
    
    cm = confusion_matrix(all_targets_mapped, all_preds, labels=np.arange(num_novel_classes))
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()
    return {"acc_unlabeled": acc}

@torch.no_grad()
def evaluate_testing_subset(model: SCKDModel, loader_test_known, loader_test_novel, 
                            Ck, Cn, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    all_preds_full, all_targets_full = [], []

    for loader in [loader_test_known, loader_test_novel]:
        for x, y_true, _ in loader:
            x = x.to(device)
            _, hl, hu = model(x)
            
            # hl đã được qua Logit Norm trong model (nếu bạn đã sửa forward)
            logits_full = torch.cat([hl, hu], dim=1) 
            pred_full = logits_full.argmax(1) 
            
            all_preds_full.extend(pred_full.cpu().tolist())
            all_targets_full.extend(y_true.tolist())

    all_preds_full = np.array(all_preds_full)
    all_targets_full = np.array(all_targets_full)

    cm = confusion_matrix(all_targets_full, all_preds_full, labels=np.arange(Ck + Cn))
    
    row_ind, col_ind = linear_sum_assignment(-cm[Ck:, Ck:])

    idx_map = {c + Ck: r + Ck for r, c in zip(row_ind, col_ind)}

    for i in range(Ck):
        idx_map[i] = i
        
    mapped_preds = np.array([idx_map.get(p, p) for p in all_preds_full])

    is_known = (all_targets_full < Ck)
    is_novel = (all_targets_full >= Ck)

    acc_known = np.mean(mapped_preds[is_known] == all_targets_full[is_known])
    acc_novel = np.mean(mapped_preds[is_novel] == all_targets_full[is_novel])
    acc_overall = np.mean(mapped_preds == all_targets_full)

    return {
        "Acc_Known_Classes": float(acc_known), 
        "Clustering_Acc_Novel_Classes": float(acc_novel), 
        "Acc_Overall": float(acc_overall)
    }
