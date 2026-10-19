"""
Apply DDI-based 3-class risk_label_eval to an existing faers_adrs.csv without re-fetching.

  adr_event=1, any ddi        -> Dangerous (risk_label_code=2)
  adr_event=0, ddi_flag=1     -> Warning   (risk_label_code=1)
  adr_event=0, ddi_flag=0     -> Safe      (risk_label_code=0)

Run: npm run ml:relabel-faers
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_faers_dataset import OUTPUT_PATH, METADATA_PATH, risk_label_code, risk_label_eval, write_csv_atomic
from data_loaders import resolve_faers_path


def main() -> None:
    source = resolve_faers_path()
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError(f"Empty dataset: {source}")

    df["ddi_flag"] = (df["ddi_pair_count"].fillna(0).astype(int) > 0).astype(int)
    df["risk_label_eval"] = df.apply(lambda row: risk_label_eval(row.to_dict()), axis=1)
    df["risk_label_code"] = df.apply(lambda row: risk_label_code(row.to_dict()), axis=1)

    target = OUTPUT_PATH if source.resolve() == OUTPUT_PATH.resolve() else source
    written = write_csv_atomic(df, target)

    distribution = df["risk_label_eval"].value_counts().to_dict()
    print(f"[relabel] Source: {source}")
    print(f"[relabel] Wrote: {written} ({len(df)} rows)")
    print(f"[relabel] risk_label_eval: {distribution}")
    print(
        f"[relabel] adr_event positive={int((df['adr_event'] == 1).sum())}, "
        f"ddi_flag={int(df['ddi_flag'].sum())}"
    )

    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    else:
        meta = {}
    meta["risk_label_eval_distribution"] = distribution
    meta["risk_label_derivation"] = (
        "adr_event=1 -> Dangerous; adr_event=0 and ddi_flag=1 -> Warning; else Safe"
    )
    meta["rows"] = int(len(df))
    meta["output"] = str(written)
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
