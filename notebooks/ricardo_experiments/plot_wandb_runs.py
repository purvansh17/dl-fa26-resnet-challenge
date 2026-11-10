import os

import matplotlib
import matplotlib.pyplot as plt
import wandb

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
})

PROJECT = "dl-resnet-2026/deeplearning-resnet-cifar10"
OUT = "figures"

FIGURES = {
    "ricardo_recipe": [
        ("Learning rate\n100 ep, [2,2,2,2], LS 0.05", [
            ("tq58t3ab", "lr 0.01"),
            ("u55fjwub", "lr 0.05"),
        ]),
        ("Label smoothing\n200 ep, lr 0.01, [2,2,2,2]", [
            ("5b4wra9u", "LS 0.05"),
            ("sci4l2j8", "LS 0.1"),
        ]),
        ("Label smoothing\n100 ep, lr 0.05, [2,2,3,1]", [
            ("c2z8rj9s", "LS 0.05"),
            ("4fh9kwh5", "LS 0.2 (run 1)"),
            ("w44eafm3", "LS 0.2 (run 2)"),
        ]),
    ],
    "ricardo_architecture": [
        ("Blocks\n100 ep, lr 0.01", [
            ("e6kph1dy", "[2,2,2,2] / [64,128,232,268]"),
            ("aidzvj68", "[4,4,4,1] / [56,112,200,256]"),
        ]),
        ("Blocks\n300 ep, lr 0.05", [
            ("vfbc3fqz", "[2,2,2,2] / [64,128,232,268]"),
            ("eopxrsgm", "[2,2,3,1] / [64,128,248,256]"),
        ]),
        ("Architecture family\n200 ep, own recipes", [
            ("5b4wra9u", "ResNet, SGD"),
            ("cqy76rh0", "ConvNeXt, AdamW"),
        ]),
    ],
    "ricardo_addons": [
        ("Regularizers\n200 ep, LS 0.1, final config", [
            ("11y38rkt", "CutMix"),
            ("y9981dmn", "SE + Cutout"),
            ("0sv4xzqp", "Degradation aug."),
        ]),
        ("EMA\n300-ep schedule, LS 0.05", [
            ("eopxrsgm", "No EMA"),
            ("etuaeof3", "EMA"),
        ]),
        ("Distillation\n200 ep, final config", [
            ("11y38rkt", "Teacher (CutMix run)"),
            ("e96qjoqu", "Logit KD student"),
            ("msim6bzl", "AT + KD student"),
        ]),
    ],
}


def load(api, run_id, cache):
    if run_id not in cache:
        run = api.run(f"{PROJECT}/{run_id}")
        hist = run.history(keys=["epoch", "val/accuracy"], pandas=True, samples=2000)
        cache[run_id] = (run, hist.sort_values("epoch"))
    return cache[run_id]


def main():
    api = wandb.Api(timeout=60)
    cache = {}
    os.makedirs(OUT, exist_ok=True)
    for name, panels in FIGURES.items():
        rows, cols = (2, 2) if len(panels) == 4 else (1, len(panels))
        fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.2 * rows), sharey=True, squeeze=False)
        for ax, (title, runs) in zip(axes.flat, panels):
            for run_id, label in runs:
                run, hist = load(api, run_id, cache)
                best = hist["val/accuracy"].max()
                line, = ax.plot(hist["epoch"], hist["val/accuracy"], label=f"{label} ({best:.2f})")
                if run.state == "crashed":
                    ax.plot(hist["epoch"].iloc[-1], hist["val/accuracy"].iloc[-1], "x",
                            color=line.get_color(), markersize=4)
                teacher = run.config.get("teacher_acc")
                if teacher and run_id == "msim6bzl":
                    ax.axhline(teacher, color=line.get_color(), linestyle="--", linewidth=0.7)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylim(85, 97.6)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
        for ax in axes[:, 0]:
            ax.set_ylabel("CIFAR-10 test acc. (%)")
        fig.tight_layout(pad=0.4, w_pad=0.6)
        fig.savefig(os.path.join(OUT, f"{name}.pdf"))
        plt.close(fig)
        print("wrote", os.path.join(OUT, f"{name}.pdf"))


if __name__ == "__main__":
    main()
