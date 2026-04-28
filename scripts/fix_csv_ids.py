#!/usr/bin/env python3
"""Fix CSV files to ensure numeric IDs are not stored in scientific notation."""

import pandas as pd
from pathlib import Path

def fix_csv_ids(csv_path: str) -> None:
    """Fix scientific notation IDs in CSV by converting to proper integers."""
    csv_path = Path(csv_path)
    
    # Read with 'id' as string
    df = pd.read_csv(csv_path, dtype={"id": str})
    
    # Ensure 'id' is truly string, trim whitespace, remove European decimal separators
    df['id'] = df['id'].str.strip()
    
    # Convert scientific notation (e.g., "1,46773E+15") to proper integers
    # First replace European decimal separator (,) with . for Python float conversion
    df['id'] = df['id'].apply(lambda x: str(int(float(x.replace(',', '.')))) if x else x)
    
    # Re-save without scientific notation
    df.to_csv(csv_path, index=False, float_format='%.0f')
    print(f"Fixed {csv_path}: {len(df)} rows processed")
    
    # Verify the fix
    sample_ids = df['id'].head(10).tolist()
    print(f"  Sample IDs after fix: {sample_ids}")


if __name__ == "__main__":
    splits_dir = Path("data/processed/splits")
    
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = splits_dir / csv_file
        if csv_path.exists():
            fix_csv_ids(str(csv_path))
        else:
            print(f"File not found: {csv_path}")
