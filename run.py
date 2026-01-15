from SCKD import *
import argparse
import os
import random

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Trả về thêm idx để ánh xạ với mảng global_labels
        img, target = self.base_dataset[self.indices[idx]]
        return img, target, idx

def get_dataloaders_train(batch_size=512, num_workers=0, name="cifar10", shuffle=True, drop_last=True):
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
        novel_classes = list(range(5, 10))
        known_classes = list(range(0, 5))
    elif name == "cifar100_20":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        novel_classes = list(range(80, 100))
        known_classes = list(range(0, 80))
    elif name == "cifar100_50":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        novel_classes = list(range(50, 100))
        known_classes = list(range(0, 50))
    elif name == "cifar100_80":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        novel_classes = list(range(20, 100))
        known_classes = list(range(0, 20))
    elif name == "imagenet_100":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        novel_classes = list(range(20, 100))
        known_classes = list(range(0, 20))

    indices_labeled = [i for i, (_, y) in enumerate(trainset) if y in known_classes]
    indices_unlabeled = [i for i, (_, y) in enumerate(trainset) if y in novel_classes]

    labeled_set = SubsetDataset(trainset, indices_labeled)
    unlabeled_set = SubsetDataset(trainset, indices_unlabeled)
    
    print(f"Labeled set: {len(labeled_set)} samples, classes: {known_classes}")
    print(f"Unlabeled set: {len(unlabeled_set)} samples, classes: {novel_classes}")

    # Kiểm tra vài samples đầu tiên
    for i in range(3):
        _, label, _ = labeled_set[i]
        print(f"Labeled sample {i}: class {label}")
    for i in range(3):
        _, label, _ = unlabeled_set[i]
        print(f"Unlabeled sample {i}: class {label}")

    # DataLoaders
    loader_labeled = DataLoader(labeled_set, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    loader_unlabeled = DataLoader(unlabeled_set, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return loader_labeled, loader_unlabeled, known_classes, novel_classes

def get_dataloaders_test(batch_size=512, num_workers=0, name="cifar100_50", train = True, shuffle=False, drop_last=False):
    # Augmentations (theo paper: crop, flip, jitter, grayscale)
    transform_test = transforms.Compose([
        transforms.ToTensor(), # Chuyển đổi thành tensor
        # Giả định ảnh 32x32 như CIFAR, không cần Resize/CenterCrop
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])
    
    if name == "cifar10":
        trainset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_test)
        novel_classes = list(range(5, 10))
        known_classes = list(range(0, 5))
    elif name == "cifar100_20":
        trainset = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform_test)
        novel_classes = list(range(80, 100))
        known_classes = list(range(0, 80))
    elif name == "cifar100_50":
        trainset = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform_test)
        novel_classes = list(range(50, 100))
        known_classes = list(range(0, 50))
    elif name == "cifar100_80":
        trainset = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform_test)
        novel_classes = list(range(20, 100))
        known_classes = list(range(0, 20))
    

    indices_labeled = [i for i, (_, y) in enumerate(trainset) if y in known_classes]
    indices_unlabeled = [i for i, (_, y) in enumerate(trainset) if y in novel_classes]

    labeled_set = SubsetDataset(trainset, indices_labeled)
    unlabeled_set = SubsetDataset(trainset, indices_unlabeled)
    
    print(f"Labeled set: {len(labeled_set)} samples, classes: {known_classes}")
    print(f"Unlabeled set: {len(unlabeled_set)} samples, classes: {novel_classes}")

    # Kiểm tra vài samples đầu tiên
    for i in range(3):
        _, label, _ = labeled_set[i]
        print(f"Labeled sample {i}: class {label}")
    for i in range(3):
        _, label, _ = unlabeled_set[i]
        print(f"Unlabeled sample {i}: class {label}")

    # DataLoaders
    loader_labeled = DataLoader(labeled_set, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    loader_unlabeled = DataLoader(unlabeled_set, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return loader_labeled, loader_unlabeled, known_classes, novel_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_stage1", type=str, default=None,
                        help="Path to Stage 1 checkpoint (skip Stage 1 if provided)")
    parser.add_argument("--resume_stage2", type=str, default=None,
                        help="Path to Stage 2 checkpoint (skip Stage 2 if provided)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    loader_labeled, loader_unlabeled, known_classes, novel_classes = get_dataloaders_train(name=args.dataset_name)
    loader_eval_labeled, loader_eval_unlabeled, _, _ = get_dataloaders_test(name=args.dataset_name, train=True, shuffle=False, drop_last=False)
    newloader_known, newloader_novel, _, _ = get_dataloaders_test(name=args.dataset_name, train=False, shuffle=False, drop_last=False)
    # 1. Config và DataLoader
    mode = "cifar"
    cl, cu = len(known_classes), len(novel_classes)
    E1, E2 = 100, 500
    
    cfg_stage1 = dict(lr=0.05, min_lr=0.001, warmup_epochs=10)
    cfg_stage2 = dict(lr=0.1, min_lr=0.005, warmup_epochs=10)

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
    else:
        print(f"\n=== Loading Stage 1 checkpoint from {args.resume_stage1} ===")

        opt1, sch1 = get_optimizer_scheduler(model, mode, cfg_stage1, total_epochs=E1)

        model = model.to(device)
        checkpoint = torch.load(args.resume_stage1, map_location=device, weights_only=False)
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
    
    # Tạo replica encoder đóng băng
    Er = clone_frozen_replica(model.encoder)
    # 5. Stage-2: discovery
    if args.resume_stage2 is None:
        print("\n=== Stage 2: Discovery Training ===")
        criterion = SCKDLoss(alpha=0.1, beta=0.1, gamma=8.0, distill_temp=1.0)
        opt2, sch2 = get_optimizer_scheduler(model, mode, cfg_stage2, total_epochs=E2)
        print("\nReplica encoder cloned and frozen")
        checkpoint = {
            'Er_state_dict': Er.state_dict()
        }
        global_pseudo_labels = None

        for e in range(E2):
            if (e + 1) > 250:
                criterion.beta = 0.005
            elif (e + 1) > 100:
                criterion.beta = 0.01  # Giảm trọng số Loss N2K sau 100 epoch
            if e == 0 or (e + 1) % 5 == 0:
                print(f"--- Updating Pseudo-labels at Epoch {e+1} ---")
                global_pseudo_labels = run_global_estep(model, loader_eval_unlabeled, cu, device)
            stats = stage2_discovery_epoch(model, Er, loader_labeled, loader_unlabeled, opt2, criterion, global_pseudo_labels, device)
            sch2.step()
            
            model.eval()
            total_known_acc = 0.0
            total_n_known = 0
            with torch.no_grad():
                for x, y, _ in loader_eval_labeled:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    _, lhl, _ = model(x)
                    pred_known = lhl.argmax(1)
                    total_known_acc += int((pred_known == y).sum().item())
                    total_n_known += y.size(0)
                stats["known_acc"] = total_known_acc / max(1, total_n_known)
            eval_stats = evaluate_training_subset(model, loader_eval_unlabeled, num_novel_classes=cu, device=device)
            stats["novel_acc"] = eval_stats["acc_unlabeled"]
            model.train()
            
            print(f"Stage-2 [Epoch {e+1:3d}/{E2}] Beta={criterion.beta} "
                f"LossTotal={stats['loss_total']:.4f} "
                f"LossCE={stats['loss_ce_total']:.4f} "
                f"LossK2N={stats['loss_k2n']:.4f} "
                f"LossN2K={stats['loss_n2k']:.4f} "
                f"LossCDC={stats['loss_cdc']:.4f} "
                f"KnownAcc={stats['known_acc']:.4f} "
                f"NovelAcc={stats['novel_acc']:.4f} ")

            if (e + 1) % 50 == 0:  # Lưu mỗi 50 epoch
                checkpoint["epoch"] = e + 1
                checkpoint["model_state_dict"] = model.state_dict()
                checkpoint["optimizer_state_dict"] = opt2.state_dict()
                checkpoint["scheduler_state_dict"] = sch2.state_dict()
                checkpoint["stats"] = stats
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage2_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                eval_stats2 = evaluate_testing_subset(model, newloader_known, newloader_novel, cl, cu, device=device)
                print("Known:", eval_stats2["Acc_Known_Classes"])
                print("Novel:", eval_stats2["Clustering_Acc_Novel_Classes"])
                print("All:", eval_stats2["Acc_Overall"])
    else:
        print(f"\n=== Loading Stage 2 checkpoint from {args.resume_stage2} ===")

        criterion = SCKDLoss(alpha=0.1, beta=0.1, gamma=8.0, distill_temp=1.0)
        opt2, sch2 = get_optimizer_scheduler(model, mode, cfg_stage2, total_epochs=E2)
        
        model = model.to(device)
        checkpoint = torch.load(args.resume_stage2, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt2.load_state_dict(checkpoint['optimizer_state_dict'])
        sch2.load_state_dict(checkpoint['scheduler_state_dict'])
        Er.load_state_dict(checkpoint['Er_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print("\nReplica encoder cloned and frozen")
        global_pseudo_labels = run_global_estep(model, loader_eval_unlabeled, cu, device)
        
        for e in range(start_epoch, E2):
            if (e + 1) > 250:
                criterion.beta = 0.005
            elif (e + 1) > 100:
                criterion.beta = 0.01  # Giảm trọng số Loss N2K sau 100 epoch
            if e == 0 or (e + 1) % 5 == 0:
                print(f"--- Updating Pseudo-labels at Epoch {e+1} ---")
                global_pseudo_labels = run_global_estep(model, loader_eval_unlabeled, cu, device)            
            stats = stage2_discovery_epoch(model, Er, loader_labeled, loader_unlabeled, opt2, criterion, global_pseudo_labels, device)
            sch2.step()
            
            model.eval()
            total_known_acc = 0.0
            total_n_known = 0
            with torch.no_grad():
                for x, y, _ in loader_eval_labeled:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    _, lhl, _ = model(x)
                    pred_known = lhl.argmax(1)
                    total_known_acc += int((pred_known == y).sum().item())
                    total_n_known += y.size(0)
                stats["known_acc"] = total_known_acc / max(1, total_n_known)
            eval_stats = evaluate_training_subset(model, loader_eval_unlabeled, num_novel_classes=cu, device=device)
            stats["novel_acc"] = eval_stats["acc_unlabeled"]
            model.train()
            
            print(f"Stage-2 [Epoch {e+1:3d}/{E2}] Beta={criterion.beta} "
                f"LossTotal={stats['loss_total']:.4f} "
                f"LossCE={stats['loss_ce_total']:.4f} "
                f"LossK2N={stats['loss_k2n']:.4f} "
                f"LossN2K={stats['loss_n2k']:.4f} "
                f"LossCDC={stats['loss_cdc']:.4f} "
                f"KnownAcc={stats['known_acc']:.4f} "
                f"NovelAcc={stats['novel_acc']:.4f} ")

            if (e + 1) % 50 == 0:  # Lưu mỗi 50 epoch
                checkpoint["epoch"] = e + 1
                checkpoint["model_state_dict"] = model.state_dict()
                checkpoint["optimizer_state_dict"] = opt2.state_dict()
                checkpoint["scheduler_state_dict"] = sch2.state_dict()
                checkpoint["stats"] = stats
                checkpoint_path = os.path.join(args.checkpoint_dir, f"stage2_epoch_{e+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                eval_stats2 = evaluate_testing_subset(model, newloader_known, newloader_novel, cl, cu, device=device)
                print("Known:", eval_stats2["Acc_Known_Classes"])
                print("Novel:", eval_stats2["Clustering_Acc_Novel_Classes"])
                print("All:", eval_stats2["Acc_Overall"])
        
    print("\n=== Training Completed ===")
    print("Model is ready for evaluation on novel classes")

    print("\n=== Evaluation on Unlabeled Training Subset ===")
    _, newloader_unlabeled, _, _ = get_dataloaders_test(name=args.dataset_name, train=True, shuffle=False, drop_last=False)
    eval_stats = evaluate_training_subset(model, newloader_unlabeled, num_novel_classes=cu, device=device)
    print("Unlabeled Clustering Acc:", eval_stats["acc_unlabeled"])
    
    print("\n=== Evaluation on Testing Subset ===")
    
    eval_stats2 = evaluate_testing_subset(model, newloader_known, newloader_novel, cl, cu, device=device)
    print("Known:", eval_stats2["Acc_Known_Classes"])
    print("Novel:", eval_stats2["Clustering_Acc_Novel_Classes"])
    print("All:", eval_stats2["Acc_Overall"])

if __name__ == "__main__":
    main()
