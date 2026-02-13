"""Plot a grid of samples from the generated circles dataset."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from generate_data import generate_batch

LABEL_COLORS = {
    0: "steelblue",
    1: "red",
    2: "orange",
    3: "green",
    4: "purple",
    5: "cyan",
}


def plot_sample(ax: plt.Axes, sample: "pd.DataFrame") -> None:
    row0 = sample.iloc[0]
    w, h = int(row0["image_w"]), int(row0["image_h"])

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for _, row in sample.iterrows():
        color = LABEL_COLORS.get(int(row["label"]), "steelblue")
        circle = mpatches.Circle(
            (row["x"], row["y"]),
            row["r"],
            fill=True,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )
        ax.add_patch(circle)

    ax.set_title(f"sample {int(row0['sample_id'])}  ({len(sample)} circles)", fontsize=8)
    ax.tick_params(labelsize=6)


def main() -> None:
    nrows, ncols = 3, 3
    num_samples = nrows * ncols

    batch = generate_batch(
        num_samples=num_samples,
        num_circles=30,
        image_size=(256, 256),
        num_labeled=3,
        max_labels=5,
        seed=42,
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

    for idx, (sample_id, group) in enumerate(batch.groupby("sample_id")):
        ax = axes[idx // ncols][idx % ncols]
        plot_sample(ax, group)

    # legend
    handles = [
        mpatches.Patch(color=LABEL_COLORS[0], label="unlabeled"),
    ]
    max_label = int(batch["label"].max())
    for lbl in range(1, max_label + 1):
        handles.append(mpatches.Patch(color=LABEL_COLORS.get(lbl, "gray"), label=f"label {lbl}"))
    fig.legend(handles=handles, loc="lower center", ncol=max_label + 1, fontsize=8)

    fig.suptitle("Generated circle samples", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig("samples.png", dpi=150)
    print("Saved samples.png")
    plt.show()


if __name__ == "__main__":
    main()
