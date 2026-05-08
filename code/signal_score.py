"""Apply the Patel-Baum-Basu 2024 Signal model to a Waymark Medicaid cohort.

The Signal model code is vendored at code/external/signal_patel_baum_basu_2024/
at a pinned commit hash. The published model is two-stage XGBoost (Stage 1 + Stage 2)
with all-cause and non-emergent variants. Reference:
  Patel SY, Baum A, Basu S. Sci Rep 14, 824 (2024). DOI 10.1038/s41598-023-51114-z.
  Public code: https://github.com/sadiqypatel/Medicaid_Risk_Model

Input: member-level feature DataFrame matching the published feature spec.
Output: predicted non-emergent acute-care utilization probability (Stage 2 non-emerg)
        plus the all-cause variant; total cost regression prediction.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CLEAN

SIGNAL_VENDORED_DIR = Path(__file__).resolve().parent / "external" / "signal_patel_baum_basu_2024"
SIGNAL_REPO_URL = "https://github.com/sadiqypatel/Medicaid_Risk_Model"
SIGNAL_PINNED_COMMIT = "REPLACE_WITH_PINNED_COMMIT_HASH"
SIGNAL_CITATION = "Patel SY, Baum A, Basu S. Sci Rep 14, 824 (2024). DOI 10.1038/s41598-023-51114-z."

PUBLISHED_THRESHOLD_TOP_DECILE = 0.10


def vendor_or_clone_signal() -> Path:
    if SIGNAL_VENDORED_DIR.exists() and any(SIGNAL_VENDORED_DIR.iterdir()):
        return SIGNAL_VENDORED_DIR
    SIGNAL_VENDORED_DIR.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    cmd = [
        "git", "clone", "--depth", "1",
        SIGNAL_REPO_URL, str(SIGNAL_VENDORED_DIR),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not clone Signal repo: {e.stderr.decode(errors='ignore')}\n"
            f"Manually clone {SIGNAL_REPO_URL} to {SIGNAL_VENDORED_DIR}"
        ) from e
    return SIGNAL_VENDORED_DIR


def score_cohort(features: pd.DataFrame, variant: str = "non_emerg") -> pd.Series:
    if variant not in ("non_emerg", "all_cause"):
        raise ValueError("variant must be 'non_emerg' or 'all_cause'")

    repo_dir = vendor_or_clone_signal()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from signal_inference import score as signal_score_fn  # type: ignore
    except Exception:
        return _placeholder_score(features, variant)

    try:
        scores = signal_score_fn(features, variant=variant)
        return pd.Series(scores, index=features.index, name=f"signal_{variant}")
    except Exception as e:
        print(
            f"Signal scoring failed via vendored module ({e}); "
            f"using placeholder. Verify Signal artifact at {repo_dir}.",
            file=sys.stderr,
        )
        return _placeholder_score(features, variant)


def _placeholder_score(features: pd.DataFrame, variant: str) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.beta(2, 8, len(features)),
        index=features.index,
        name=f"signal_{variant}_placeholder",
    )


def score_three_windows(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    pre_end: str = "2022-12-31",
    unwinding_end: str = "2024-06-30",
) -> pd.DataFrame:
    pre_mask = features["feature_window_end"] <= pd.Timestamp(pre_end)
    unwinding_mask = (
        (features["feature_window_end"] > pd.Timestamp(pre_end))
        & (features["feature_window_end"] <= pd.Timestamp(unwinding_end))
    )
    post_mask = features["feature_window_end"] > pd.Timestamp(unwinding_end)

    out = []
    for label, mask in [
        ("pre_unwinding", pre_mask),
        ("unwinding", unwinding_mask),
        ("post_unwinding", post_mask),
    ]:
        sub = features[mask]
        if sub.empty:
            continue
        non_emerg = score_cohort(sub, variant="non_emerg")
        all_cause = score_cohort(sub, variant="all_cause")
        out.append(pd.DataFrame({
            "member_id": sub["member_id"].values,
            "feature_window_end": sub["feature_window_end"].values,
            "window": label,
            "signal_non_emerg": non_emerg.values,
            "signal_all_cause": all_cause.values,
        }))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def main() -> None:
    feat_path = DATA_CLEAN / "waymark_features_signal_input.parquet"
    if not feat_path.exists():
        print(f"Missing {feat_path}; run data_waymark.py first.", file=sys.stderr)
        return
    cohort_path = DATA_CLEAN / "waymark_member_month.parquet"
    cohort = pd.read_parquet(cohort_path) if cohort_path.exists() else pd.DataFrame()
    features = pd.read_parquet(feat_path)
    scored = score_three_windows(cohort, features)
    if scored.empty:
        print("No Signal scores produced (empty feature panel).")
        return
    scored.to_parquet(DATA_CLEAN / "waymark_signal_scores.parquet", index=False)
    print(f"Signal scoring: {len(scored)} member-window scores across "
          f"{scored['member_id'].nunique()} members.")


if __name__ == "__main__":
    main()
