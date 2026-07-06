from __future__ import annotations

from pathlib import Path

import pandas as pd


# 0.1 CSV Output
def save_output(df: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    df.to_csv(file_path, index=False)
    return file_path


def load_output(output_dir: str | Path, filename: str, **kwargs) -> pd.DataFrame:
    file_path = Path(output_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing output file: {file_path}")
    return pd.read_csv(file_path, **kwargs)
