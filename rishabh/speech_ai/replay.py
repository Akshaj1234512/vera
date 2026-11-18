"""Write a compact log and plot for a paired audio replay."""

import csv
import json
from pathlib import Path


def write_timeline(timeline: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "timeline.json").write_text(json.dumps(timeline, indent=2, allow_nan=False) + "\n")
    with (output / "cues.csv").open("w", newline="") as file:
        fields = ["onset_s", "duration_s", "role", "cue", "target", "reason"]
        rows = ({**event, "target": json.dumps(event["target"])} for event in timeline["events"])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_timeline(timeline: dict, path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    time = np.array(timeline["time_s"])
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1.4, 1.2]},
        layout="constrained",
    )
    fig.set_facecolor("#f6f8fa")
    fig.suptitle(f'{timeline["source"]}\n{timeline["policy"]} / {timeline["profile"]}',
                 x=0.04, ha="left", fontsize=15, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        for item in timeline["overlap_cessations"]:
            ax.axvspan(item["overlap_onset_s"], item["agent_cessation_s"], color="#fed7aa", alpha=0.6)
    for ax, track, color, label in zip(
        axes[:2], ("partner", "agent"), ("#4338ca", "#03695d"), ("Partner", "Agent")
    ):
        ax.plot(time, timeline["feature_db"][track], color=color, linewidth=1.25)
        ax.set_ylim(-82, 2)
        ax.set_yticks([-70, -35, 0])
        ax.set_ylabel(f"{label}\ndBFS")
    axes[2].plot(time, timeline["jaw_aperture"], color="#03695d", linewidth=1.5)
    axes[2].set(ylim=(-0.03, 1.04), yticks=[0, 1], ylabel="Jaw\ntarget")
    cue_colors = {"nod": "#4338ca", "gaze": "#7c3aed", "settle": "#b45309"}
    present = set()
    for event in timeline["events"]:
        cue = event["cue"]
        present.add(cue)
        axes[3].axvspan(event["onset_s"], event["onset_s"] + event["duration_s"],
                        ymin=0.2, ymax=0.8, color=cue_colors[cue], alpha=0.84)
    for decision in timeline["ack_decisions"]:
        if decision["cue"] is None:
            axes[3].axvline(decision["time_s"], ymin=0.25, ymax=0.75, color="#b0b8c2", alpha=0.6)
    handles = [Patch(facecolor=cue_colors[cue], label=cue) for cue in cue_colors if cue in present]
    if timeline["overlap_cessations"]:
        handles.append(Patch(facecolor="#fed7aa", label="detected overlap before cessation"))
    if handles:
        axes[3].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.93)
    axes[3].set(xlabel="Audio timeline (s)", xlim=(0, time[-1]), yticks=[], ylabel="Cues")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
