import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torchsummary import summary
from tqdm import tqdm


class LayerNorm2d(nn.LayerNorm):

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):

    def __init__(self, dim, kernel_size=7, drop_path=0.0, layer_scale_init=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = drop_path

    def forward(self, x):
        out = self.dwconv(x)
        out = out.permute(0, 2, 3, 1)
        out = self.pwconv2(F.gelu(self.pwconv1(self.norm(out))))
        out = (self.gamma * out).permute(0, 3, 1, 2)

        if self.training and self.drop_path > 0:
            keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_path
            out = out * keep / (1 - self.drop_path)
        return x + out


class ConvNeXt(nn.Module):

    def __init__(self, depths, dims, kernel_size=7, drop_path=0.1, num_classes=10):
        super(ConvNeXt, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0]),
        )

        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        idx = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            if i > 0:
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2),
                ))
            self.stages.append(nn.Sequential(
                *[ConvNeXtBlock(dim, kernel_size, rates[idx + j]) for j in range(depth)]
            ))
            idx += depth

        self.norm = nn.LayerNorm(dims[-1])
        self.linear = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                out = self.downsamples[i - 1](out)
            out = stage(out)
        out = out.mean(dim=(2, 3))
        return self.linear(self.norm(out))


def make_optimizer(model, config):
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    return optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"],
    )


def train_convnext(model, train_dataloader, val_dataloader, criterion, optimizer, config, epochs=10,
          device="cuda", checkpoint="best_convnext.pth"):
    model = model.to(device)

    warmup_epochs = config.get("warmup_epochs", 5)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

    wandb.init(project="deeplearning-resnet-cifar10", config=config)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        with tqdm(total=len(train_dataloader), desc=f"Train Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(
                    {
                        "Loss": f"{train_loss/(batch_idx+1):.3f}",
                        "Acc": f"{100.0 * correct / total:.3f}%",
                    }
                )
                pbar.update(1)

        epoch_loss = train_loss / len(train_dataloader)
        epoch_acc = 100.0 * correct / total

        val_acc = evaluate(model, val_dataloader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": epoch_loss,
            "train/accuracy": epoch_acc,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_accuracy": best_val_acc,
        })

        print(
            f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    wandb.finish()

    return model


def run_convnext(config, train_dataloader, val_dataloader, device="cuda", checkpoint="best_convnext.pth"):
    model = ConvNeXt(depths=config["depths"], dims=config["dims"], drop_path=config["drop_path"]).to(device)
    summary(model, (3, 32, 32), device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    assert num_params <= 5_000_000

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = make_optimizer(model, config)
    return train_convnext(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        epochs=config["epochs"],
        device=device,
        checkpoint=checkpoint,
    )
