"""Train and compare linear and GRU pause scorers with speaker-group holdouts."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .audio import FEATURE_NAMES, HISTORY_FRAMES, Features, aligned_embeddings, context_windows
from .cues import RULE_THRESHOLD, quiet_agent, rule_scores, selected_candidates
from .dataset import Clip, load_manifest, match_events, split_groups

MODEL_VERSION = 1


class PauseScorer(nn.Module):
    def __init__(self, architecture: str, embedding_dim: int):
        super().__init__()
        self.sequence = (nn.GRU(len(FEATURE_NAMES), 24, batch_first=True)
                         if architecture == "gru" else None)
        width = 24 if self.sequence is not None else HISTORY_FRAMES * len(FEATURE_NAMES)
        self.output = nn.Linear(width + embedding_dim, 1)

    def forward(self, windows: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if self.sequence is None:
            acoustic = windows.flatten(1)
        else:
            _, state = self.sequence(windows)
            acoustic = state[-1]
        return self.output(torch.cat([acoustic, embeddings], dim=1)).squeeze(1)


def labels_for_candidates(clip: Clip, tolerance: float) -> np.ndarray:
    features = clip.features
    available = np.flatnonzero([quiet_agent(features, index) for index in features.candidates])
    times = features.time_s[features.candidates[available]]
    labels = np.zeros(len(features.candidates), dtype=np.float32)
    for p, _ in match_events(times, clip.labels, tolerance):
        labels[available[p]] = 1
    return labels


def embedding_inputs(splits: dict[str, list[Clip]]) -> dict[str, np.ndarray]:
    aligned = {}
    for clips in splits.values():
        for clip in clips:
            if clip.embedding_path is None:
                raise ValueError(f"{clip.id}: --embeddings requires a causal embeddings file for every clip")
            aligned[clip.id] = aligned_embeddings(
                clip.embedding_path, clip.features.time_s[clip.features.candidates]
            )
    widths = {array.shape[1] for array in aligned.values()}
    if len(widths) != 1:
        raise ValueError("All embeddings must have the same dimension")
    for name, clips in splits.items():
        if not any(aligned[clip.id][:, -1].any() for clip in clips):
            raise ValueError(f"{name} has no fresh embeddings at acknowledgment candidates")
    return aligned


def normalize_embeddings(array: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    features, mask = array[:, :-1], array[:, -1:]
    return torch.cat([((features - mean) / std) * mask, mask], dim=1)


def metrics(clips: list[Clip], scores: dict[str, np.ndarray], threshold: float, tolerance: float) -> dict:
    true = false = missed = reachable = 0
    duration = 0.0
    timing_errors = []
    for clip in clips:
        features = clip.features
        selected = selected_candidates(features, scores[clip.id], threshold)
        times = features.time_s[selected]
        matches = match_events(times, clip.labels, tolerance)
        true += len(matches)
        false += len(times) - len(matches)
        missed += len(clip.labels) - len(matches)
        reachable += int(labels_for_candidates(clip, tolerance).sum())
        duration += features.time_s[-1]
        timing_errors += [abs(times[p] - clip.labels[y]) for p, y in matches]
    precision = true / max(1, true + false)
    recall = true / max(1, true + missed)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "tp": true, "fp": false, "fn": missed,
        "pause_candidate_recall_ceiling": round(reachable / max(1, true + missed), 4),
        "false_cues_per_min": round(60 * false / duration, 3),
        "mean_abs_onset_error_ms": round(float(np.mean(timing_errors)) * 1000, 1) if timing_errors else None,
    }


def score_clips(model: PauseScorer, clips: list[Clip], embeddings: dict[str, np.ndarray],
                mean: torch.Tensor, std: torch.Tensor) -> dict[str, np.ndarray]:
    scores = {}
    with torch.inference_mode():
        for clip in clips:
            history = torch.from_numpy(context_windows(clip.features))
            states = torch.from_numpy(embeddings.get(clip.id, np.empty((len(history), 0), dtype=np.float32)))
            if states.shape[1]:
                states = normalize_embeddings(states, mean, std)
            scores[clip.id] = (model(history, states).sigmoid().numpy()
                               if len(history) else np.empty(0, dtype=np.float32))
    return scores


def fit(architecture: str, splits: dict[str, list[Clip]], embeddings: dict[str, np.ndarray],
        epochs: int, seed: int, tolerance: float, vad_db: float, output: Path) -> dict:
    clips = splits["train"]
    windows = torch.from_numpy(np.concatenate([context_windows(clip.features) for clip in clips]))
    labels = torch.from_numpy(np.concatenate([labels_for_candidates(clip, tolerance) for clip in clips]))
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if min(positives, negatives) == 0:
        raise ValueError("Train groups need both positive and negative pause candidates")
    if embeddings:
        states = torch.from_numpy(np.concatenate([embeddings[clip.id] for clip in clips]))
        present = states[states[:, -1] > 0, :-1]
        mean = present.mean(dim=0)
        std = present.std(dim=0, correction=0).clamp_min(1e-5)
        states = normalize_embeddings(states, mean, std)
    else:
        states = torch.empty((len(labels), 0))
        mean = std = torch.empty(0)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = PauseScorer(architecture, states.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    for _ in range(epochs):
        for indices in torch.randperm(len(labels), generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model(windows[indices], states[indices]), labels[indices])
            objective.backward()
            optimizer.step()

    model.eval()
    validation = splits["validation"]
    val_scores = score_clips(model, validation, embeddings, mean, std)
    candidates = []
    for threshold in np.linspace(0.05, 0.99, 48):
        result = metrics(validation, val_scores, float(threshold), tolerance)
        candidates.append((result["f1"], -result["fp"], float(threshold), result))
    _, _, threshold, val_result = max(candidates, key=lambda item: item[:3])

    test = splits["test"]
    test_scores = score_clips(model, test, embeddings, mean, std)
    with torch.inference_mode():
        history = windows[:1]
        state = states[:1]
        for _ in range(5):
            model(history, state)
        started = perf_counter()
        for _ in range(100):
            model(history, state)
        ms_per_candidate = 10 * (perf_counter() - started)
    name = architecture + ("-embeddings" if embeddings else "")
    torch.save({
        "version": MODEL_VERSION, "architecture": architecture, "embedding_dim": states.shape[1],
        "embedding_mean": mean, "embedding_std": std, "threshold": threshold, "vad_db": vad_db,
        "weights": model.state_dict(),
    }, output / f"{name}.pt")
    return {
        "name": name, "threshold": round(threshold, 4), "parameters": sum(p.numel() for p in model.parameters()),
        "cpu_ms_per_candidate": round(ms_per_candidate, 4),
        "validation": val_result, "test": metrics(test, test_scores, threshold, tolerance),
    }


def audit(manifest: Path, vad_db: float, seed: int, tolerance: float) -> dict:
    clips = load_manifest(manifest, vad_db)
    splits = split_groups(clips, seed)
    results = {
        "origins": sorted({clip.origin for clip in clips}),
        "vad_db": vad_db, "match_tolerance_s": tolerance,
        "splits": {},
    }
    for name, portions in splits.items():
        rows = []
        for clip in portions:
            features = clip.features
            times = features.time_s[features.candidates]
            rows.append({
                "id": clip.id, "group": clip.group,
                "labels": len(clip.labels), "pause_candidates": len(times),
                "matches_before_speech_gate": len(match_events(times, clip.labels, tolerance)),
                "matches_after_speech_gate": int(labels_for_candidates(clip, tolerance).sum()),
            })
        results["splits"][name] = rows
    return results


def train(manifest: Path, output: Path, epochs: int, seed: int, tolerance: float,
          vad_db: float, use_embeddings: bool) -> dict:
    if epochs <= 0 or tolerance < 0:
        raise ValueError("Use positive epochs and a nonnegative matching tolerance")
    torch.set_num_threads(1)
    clips = load_manifest(manifest, vad_db)
    splits = split_groups(clips, seed)
    test = splits["test"]
    output.mkdir(parents=True, exist_ok=True)
    rules = metrics(test, {clip.id: rule_scores(clip.features) for clip in test}, RULE_THRESHOLD, tolerance)
    configurations = [{}]
    if use_embeddings:
        configurations.append(embedding_inputs(splits))
    models = [fit(architecture, splits, embeddings, epochs, seed, tolerance, vad_db, output)
              for embeddings in configurations for architecture in ("linear", "gru")]
    report = {
        "origins": sorted({clip.origin for clip in clips}), "seed": seed,
        "epochs": epochs, "match_tolerance_s": tolerance, "vad_db": vad_db,
        "splits": {
            name: {
                "clips": [clip.id for clip in portions], "groups": sorted({clip.group for clip in portions}),
                "annotated_events": sum(len(clip.labels) for clip in portions),
            } for name, portions in splits.items()
        },
        "rules_test": rules, "models": models,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def checkpoint_scores(path: Path, features: Features, embedding_path: Path | None,
                      vad_db: float) -> tuple[np.ndarray, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["version"] != MODEL_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version")
    if not np.isclose(vad_db, checkpoint["vad_db"]):
        raise ValueError(f"Use --vad-db {checkpoint['vad_db']} to match the checkpoint")
    model = PauseScorer(checkpoint["architecture"], checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    clip = Clip("replay", "", "", np.empty(0), features, embedding_path)
    embeddings = {}
    if checkpoint["embedding_dim"]:
        if embedding_path is None:
            raise ValueError("Checkpoint requires --embeddings")
        array = aligned_embeddings(embedding_path, features.time_s[features.candidates])
        if array.shape[1] != checkpoint["embedding_dim"]:
            raise ValueError("Replay embedding dimension differs from the checkpoint")
        embeddings[clip.id] = array
    elif embedding_path is not None:
        raise ValueError("This checkpoint uses acoustic features only")
    scores = score_clips(model, [clip], embeddings, checkpoint["embedding_mean"], checkpoint["embedding_std"])
    return scores[clip.id], checkpoint["threshold"]
