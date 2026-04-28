#!/usr/bin/env python3
"""Fix CSVs by removing rows with null misinformation labels."""

import pandas as pd
from pathlib import Path

def fix_csv_remove_null_labels(csv_path: str) -> None:
    """Remove rows with null misinformation labels from CSV."""
    csv_path = Path(csv_path)
    
    # Read with 'id' as string
    df = pd.read_csv(csv_path, dtype={"id": str})
    
    # Count before
    n_before = len(df)
    
    # Drop rows with null misinformation
    df = df.dropna(subset=["misinformation"]).reset_index(drop=True)
    
    # Count after
    n_after = len(df)
    n_removed = n_before - n_after
    
    # Re-save
    df.to_csv(csv_path, index=False, float_format='%.0f')
    print(f"Fixed {csv_path}: Removed {n_removed} rows with null labels ({n_before} → {n_after})")


if __name__ == "__main__":
    splits_dir = Path("data/processed/splits")
    
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = splits_dir / csv_file
        if csv_path.exists():
            fix_csv_remove_null_labels(str(csv_path))
        else:
            print(f"File not found: {csv_path}")
