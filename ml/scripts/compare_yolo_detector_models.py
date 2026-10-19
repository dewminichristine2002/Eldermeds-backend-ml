from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def _clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_final_metrics(results_csv: Path) -> dict[str, Any]:
    with results_csv.open(newline="", encoding="utf-8") as file:
        rows = [_clean_row(row) for row in csv.DictReader(file)]

    if not rows:
        raise ValueError(f"No metric rows found in {results_csv}")

    final_row = rows[-1]
    metrics: dict[str, Any] = {"epochs_completed": int(float(final_row.get("epoch", 0)))}
    for field in METRIC_FIELDS:
        metrics[field] = _float_value(final_row, field)
    return metrics


def train_model(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from ultralytics import YOLO

    model_label = Path(model_name).stem
    run_name = f"{args.name_prefix}_{model_label}"
    run_dir = args.project / run_name
    results_csv = run_dir / "results.csv"

    if args.skip_existing and results_csv.exists():
        duration_seconds = math.nan
    else:
        started_at = time.perf_counter()
        model = YOLO(model_name)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            project=str(args.project),
            name=run_name,
            exist_ok=args.exist_ok,
            workers=args.workers,
            patience=args.patience,
            plots=args.plots,
            seed=args.seed,
            deterministic=True,
            fraction=args.fraction,
        )
        duration_seconds = time.perf_counter() - started_at

    metrics = read_final_metrics(results_csv)
    weights_path = run_dir / "weights" / "best.pt"
    size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else math.nan

    return {
        "model": Path(model_name).name,
        "run": run_name,
        "results_csv": str(results_csv),
        "best_weights": str(weights_path),
        "best_weights_mb": size_mb,
        "train_seconds": duration_seconds,
        **metrics,
    }


def choose_deployment_model(rows: list[dict[str, Any]], map_tolerance: float) -> dict[str, Any]:
    valid_rows = [row for row in rows if not math.isnan(row["metrics/mAP50-95(B)"])]
    if not valid_rows:
        raise ValueError("No valid mAP50-95 results found.")

    for row in valid_rows:
        size_mb = row["best_weights_mb"]
        row["deployment_score"] = (
            row["metrics/mAP50-95(B)"] / size_mb
            if size_mb and not math.isnan(size_mb)
            else math.nan
        )

    best_score = max(row["deployment_score"] for row in valid_rows)
    comparable_rows = [
        row for row in valid_rows if best_score - row["deployment_score"] <= map_tolerance
    ]
    return sorted(comparable_rows, key=lambda row: (row["best_weights_mb"], row["model"]))[0]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.5f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "run",
        "epochs_completed",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "best_weights_mb",
        "deployment_score",
        "train_seconds",
        "results_csv",
        "best_weights",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fieldnames})


def write_markdown(
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    output_md: Path,
    map_tolerance: float,
    training_fraction: float,
) -> None:
    lines = [
        "# YOLO Pill Detector Model Comparison",
        "",
        "Each model was trained on the same pill detector dataset and training settings.",
        f"Training fraction: {training_fraction:.3f}",
        "The deployment model is selected by mAP50-95 per MB, which balances validation accuracy with model size.",
        "",
        f"Selected deployment model: **{selected['model']}**",
        "",
        f"Selection rule: highest deployment score, treating scores within {map_tolerance:.3f} as comparable and selecting the smaller model.",
        "",
        "| Model | Precision | Recall | mAP50 | mAP50-95 | Weights MB | Deployment Score | Epochs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in sorted(rows, key=lambda item: item["model"]):
        lines.append(
            "| {model} | {precision} | {recall} | {map50} | {map5095} | {size} | {score} | {epochs} |".format(
                model=row["model"],
                precision=_format_value(row["metrics/precision(B)"]),
                recall=_format_value(row["metrics/recall(B)"]),
                map50=_format_value(row["metrics/mAP50(B)"]),
                map5095=_format_value(row["metrics/mAP50-95(B)"]),
                size=_format_value(row["best_weights_mb"]),
                score=_format_value(row["deployment_score"]),
                epochs=row["epochs_completed"],
            )
        )

    lines.extend(
        [
            "",
            "Evidence files:",
            "",
        ]
    )
    for row in sorted(rows, key=lambda item: item["model"]):
        lines.append(f"- {row['model']}: `{row['results_csv']}`")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiple YOLO pill detector models and summarize their validation metrics."
    )
    parser.add_argument("--data", type=Path, default=Path("pill-detector-data.yaml"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        help="YOLO checkpoints to compare.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name-prefix", default="pill_detector_compare")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use for each model.",
    )
    parser.add_argument("--map-tolerance", type=float, default=0.01)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/detect/model_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/detect/model_comparison.md"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    rows = [train_model(args, model_name) for model_name in args.models]
    selected = choose_deployment_model(rows, args.map_tolerance)

    write_csv(rows, args.output_csv)
    write_markdown(rows, selected, args.output_md, args.map_tolerance, args.fraction)

    print(f"Wrote comparison CSV: {args.output_csv}")
    print(f"Wrote comparison report: {args.output_md}")
    print(f"Selected deployment model: {selected['model']}")


if __name__ == "__main__":
    main()
