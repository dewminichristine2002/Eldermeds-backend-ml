"""Render Stage 7A audit figures from the generated machine-readable files."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent / "20261008T_STAGE7A_FINAL"
audit = json.loads((RUN / "dataset_quality_inventory.json").read_text(encoding="utf-8"))
development = next(item for item in audit["datasets"] if item["file"] == "development_dataset_v3.csv")
labels = list(development["label_counts"])
counts = [development["label_counts"][label] for label in labels]
with (RUN / "annotation_review_queue.csv").open(encoding="utf-8-sig", newline="") as stream:
    queue_counts = Counter(row["reason"] for row in csv.DictReader(stream))

figure, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(labels, counts, color="#4677a8")
axes[0].set_title("Development corpus class counts")
axes[0].set_ylabel("Rows")
axes[0].tick_params(axis="x", rotation=35)
axes[1].barh(list(queue_counts), list(queue_counts.values()), color="#c58145")
axes[1].set_title("Stage 7A review queue reasons (n=500)")
axes[1].set_xlabel("Rows")
figure.tight_layout()
figure.savefig(RUN / "stage7a_quality_figures.png", dpi=160)
plt.close(figure)
