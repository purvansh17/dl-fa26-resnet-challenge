def kd_loss(student_logits, teacher_logits, temperature):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature ** 2


def features_and_logits(model, x):
    out = F.relu(model.bn1(model.conv1(x)))
    feats = []
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        out = layer(out)
        feats.append(out)
    out = F.avg_pool2d(out, model.avg_pooling_size).flatten(1)
    return feats, model.linear(out)


def attention_map(f):
    return F.normalize(f.pow(2).mean(1).flatten(1), dim=1)


def at_loss(student_feats, teacher_feats):
    return sum((attention_map(s) - attention_map(t)).pow(2).mean()
               for s, t in zip(student_feats, teacher_feats))


def train_distill(student, teacher, train_dataloader, val_dataloader, optimizer, epochs=300,
                  temperature=4.0, kd_weight=0.9, at_weight=1000.0, label_smoothing=0.05,
                  device="cuda", checkpoint="best_distill.pth"):
    student = student.to(device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("Teacher accuracy on val:")
    teacher_acc = evaluate(teacher, val_dataloader, device)

    run = wandb.init(project="deeplearning-resnet-cifar10",
                     config={**config, "teacher_acc": teacher_acc, "temperature": temperature,
                             "kd_weight": kd_weight, "at_weight": at_weight})
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        student.train()
        train_loss = 0.0
        correct = 0
        total = 0

        with tqdm(total=len(train_dataloader), desc=f"Train Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.no_grad():
                    t_feats, t_logits = features_and_logits(teacher, inputs)
                    t_logits = (t_logits + teacher(torch.flip(inputs, dims=[3]))) / 2

                optimizer.zero_grad()
                s_feats, outputs = features_and_logits(student, inputs)
                hard = ce(outputs, targets)
                soft = kd_loss(outputs, t_logits, temperature)
                hidden = at_loss(s_feats, t_feats)
                loss = kd_weight * soft + (1 - kd_weight) * hard + at_weight * hidden
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total += targets.size(0)
                correct += outputs.argmax(1).eq(targets).sum().item()

                pbar.set_postfix({"Loss": f"{train_loss/(batch_idx+1):.3f}",
                                  "AT": f"{at_weight * hidden.item():.3f}",
                                  "Acc": f"{100.0 * correct / total:.3f}%"})
                pbar.update(1)

        val_acc = evaluate(student, val_dataloader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), checkpoint)

        scheduler.step()
        run.log({
            "epoch": epoch + 1,
            "train/loss": train_loss / len(train_dataloader),
            "train/accuracy": 100.0 * correct / total,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_accuracy": best_val_acc,
        })

    artifact = wandb.Artifact(name=f"resnet-distill-at-{run.id}", type="model",
                              metadata={"best_val_accuracy": best_val_acc, "teacher_acc": teacher_acc})
    artifact.add_file(checkpoint)
    run.log_artifact(artifact)
    run.finish()

    return student


teacher = ResNet5M(blocks=config["blocks"], channels=config["channels"])
sd = torch.load("best-last.pth", map_location=device, weights_only=True)
teacher.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})

student = ResNet5M(blocks=config["blocks"], channels=config["channels"], kaiming=config["kaiming"]).to(device)
optimizer = optim.SGD(student.parameters(), lr=config["lr"], momentum=config["momentum"],
                      weight_decay=config["weight_decay"], nesterov=True)

student = train_distill(
    student, teacher,
    train_dataloader=dataloader['train'],
    val_dataloader=CIFAROriginal(batch_size=config["batch_size"], num_workers=config["workers"]).test_loader,
    optimizer=optimizer,
    epochs=config["epochs"],
    temperature=4.0, kd_weight=0.9, at_weight=1000.0,
)
