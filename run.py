from SCKD import *
import argparse
import os

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

def get_dataloaders(batch_size=512, num_workers=4, name="cifar10"):
    # Augmentations (theo paper: crop, flip, jitter, grayscale)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    if name == "cifar10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        # Chia lớp: 0–4 = known, 5–9 = novel
        known_classes = list(range(0, 5))
        novel_classes = list(range(5, 10))
    elif name == "cifar100_20":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        # Chia lớp: 0–79 = known, 80–99 = novel
        known_classes = list(range(0, 80))
        novel_classes = list(range(80, 100))
    elif name == "cifar100_50":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        # Chia lớp: 0–49 = known, 50–99 = novel
        known_classes = list(range(0, 50))
        novel_classes = list(range(50, 100))
    

    indices_labeled = [i for i, (_, y) in enumerate(trainset) if y in known_classes]
    indices_unlabeled = [i for i, (_, y) in enumerate(trainset) if y in novel_classes]

    labeled_set = SubsetDataset(trainset, indices_labeled)
    unlabeled_set = SubsetDataset(trainset, indices_unlabeled)
    
    print(f"Labeled set: {len(labeled_set)} samples, classes: {known_classes}")
    print(f"Unlabeled set: {len(unlabeled_set)} samples, classes: {novel_classes}")

    # Kiểm tra vài samples đầu tiên
    for i in range(3):
        _, label = labeled_set[i]
        print(f"Labeled sample {i}: class {label}")
    for i in range(3):
        _, label = unlabeled_set[i]
        print(f"Unlabeled sample {i}: class {label}")

    # DataLoaders
    loader_labeled = DataLoader(labeled_set, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, drop_last=True)
    loader_unlabeled = DataLoader(unlabeled_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)

    return loader_labeled, loader_unlabeled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_stage1", type=str, default=None,
                        help="Path to Stage 1 checkpoint (skip Stage 1 if provided)")
    parser.add_argument("--resume_stage2", type=str, default=None,
                        help="Path to Stage 2 checkpoint (skip Stage 2 if provided)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Config và DataLoader
    mode = "cifar"
    cl, cu = 5, 5
    E1, E2 = 100, 500
    
    cfg_stage1 = dict(lr=0.4, min_lr=1e-3, warmup_epochs=10)
    cfg_stage2 = dict(lr=0.4, min_lr=1e-3, warmup_epochs=10)

    loader_labeled, loader_unlabeled = get_dataloaders(name="cifar10")

    # 2. Build model
    model = build_model(mode, cl, cu, device)
    print(f"Model built with {cl} known classes and {cu} novel classes")

    # 3. Stage-1: supervised
    if args.resume_stage1 is None:
        print("\n=== Stage 1: Supervised Training ===")
        opt1, sch1 = get_optimizer_scheduler(model, mode, cfg_stage1, total_epochs=E1)
        
        for e in range(E1):
            stats = stage1_supervised_epoch(model, loader_labeled, opt1, device)
            sch1.step()
            
            print(f"Stage-1 [Epoch {e+1:3d}/{E1}] "
                    f"Loss={stats['sup_loss']:.4f} "
                    f"Acc={stats['sup_acc']:.4f}")
            if (e + 1) % 10 == 0:
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt1.state_dict(),
                    'scheduler_state_dict': sch1.state_dict(),
                    'stats': stats
                }
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage1_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "stage1_final.pth"))
    else:
        print(f"\n=== Loading Stage 1 checkpoint from {args.resume_stage1} ===")
        
        opt1, sch1 = get_optimizer_scheduler(model, mode, cfg_stage1, total_epochs=E1)
        
        model = model.to(device)
        checkpoint = torch.load(args.resume_stage1, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt1.load_state_dict(checkpoint['optimizer_state_dict'])
        sch1.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        
        for e in range(start_epoch, E1):
            stats = stage1_supervised_epoch(model, loader_labeled, opt1, device)
            sch1.step()
            
            print(f"Stage-1 [Epoch {e+1:3d}/{E1}] "
                    f"Loss={stats['sup_loss']:.4f} "
                    f"Acc={stats['sup_acc']:.4f}")
            if (e + 1) % 10 == 0:
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt1.state_dict(),
                    'scheduler_state_dict': sch1.state_dict(),
                    'stats': stats
                }
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage1_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "stage1_final.pth"))

    # 5. Stage-2: discovery
    if args.resume_stage2 is None:
        print("\n=== Stage 2: Discovery Training ===")
        criterion = SCKDLoss(alpha=0.1, beta=0.5, distill_temp=1.0, sinkhorn_iter=5, sinkhorn_eps=0.35)
        opt2, sch2 = get_optimizer_scheduler(model, mode, cfg_stage2, total_epochs=E2)
        
        for e in range(E2):
            Er = clone_frozen_replica(model.encoder)
            print("\nReplica encoder cloned and frozen")
            stats = stage2_discovery_epoch(model, Er, loader_labeled, loader_unlabeled, opt2, criterion, device)
            sch2.step()
            
            print(f"Stage-2 [Epoch {e+1:3d}/{E2}] "
                f"Total={stats['loss_total']:.4f} "
                f"CE={stats['loss_ce']:.4f} "
                f"Sup={stats['loss_sup']:.4f} "
                f"Unl={stats['loss_unl']:.4f} "
                f"K2N={stats['loss_k2n']:.4f} "
                f"N2K={stats['loss_n2k']:.4f}")
            
            if (e + 1) % 50 == 0:  # Lưu mỗi 50 epoch
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt2.state_dict(),
                    'scheduler_state_dict': sch2.state_dict(),
                    'stats': stats,
                    'Er_state_dict': Er.state_dict()  # Lưu cả replica encoder
                }
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage2_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "stage2_final.pth"))
    else:
        print(f"\n=== Loading Stage 2 checkpoint from {args.resume_stage2} ===")

        criterion = SCKDLoss(alpha=0.1, beta=0.5, distill_temp=1.0, sinkhorn_iter=5, sinkhorn_eps=0.35)
        opt2, sch2 = get_optimizer_scheduler(model, mode, cfg_stage2, total_epochs=E2)
        
        model = model.to(device)
        checkpoint = torch.load(args.resume_stage2, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt2.load_state_dict(checkpoint['optimizer_state_dict'])
        sch2.load_state_dict(checkpoint['scheduler_state_dict'])
        Er = clone_frozen_replica(model.encoder)
        
        # for p in Er.parameters():
        #     p.requires_grad = False
        
        start_epoch = checkpoint['epoch']
        
        for e in range(start_epoch, E2):
            if e != start_epoch:
                Er = clone_frozen_replica(model.encoder)
            print("\nReplica encoder cloned and frozen")
            stats = stage2_discovery_epoch(model, Er, loader_labeled, loader_unlabeled, opt2, criterion, device)
            sch2.step()
            
            print(f"Stage-2 [Epoch {e+1:3d}/{E2}] "
                f"Total={stats['loss_total']:.4f} "
                f"CE={stats['loss_ce']:.4f} "
                f"Sup={stats['loss_sup']:.4f} "
                f"Unl={stats['loss_unl']:.4f} "
                f"K2N={stats['loss_k2n']:.4f} "
                f"N2K={stats['loss_n2k']:.4f}")
            
            if (e + 1) % 50 == 0:  # Lưu mỗi 50 epoch
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt2.state_dict(),
                    'scheduler_state_dict': sch2.state_dict(),
                    'stats': stats,
                    'Er_state_dict': Er.state_dict()  # Lưu cả replica encoder
                }
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage2_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "stage2_final.pth"))
        
    print("\n=== Training Completed ===")
    print("Model is ready for evaluation on novel classes")
    # -------- Evaluation --------
    print("\n=== Evaluation on Unlabeled Subset ===")
    _, newloader_unlabeled = get_dataloaders()
    eval_stats = evaluate_task_aware(model, newloader_unlabeled, num_novel_classes=cu, device=device)
    print("Unlabeled Clustering Acc:", eval_stats["acc_unlabeled"])
        
if __name__ == "__main__":
    main()