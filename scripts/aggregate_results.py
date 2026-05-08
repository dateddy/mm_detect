#!/usr/bin/env python3
"""
Aggregate ablation results into paper-ready tables.

Usage:
    # Aggregate all completed ablations
    python scripts/aggregate_results.py
    
    # Specify custom results directory
    python scripts/aggregate_results.py --results-dir outputs/
    
    # Output formats
    python scripts/aggregate_results.py --format markdown
    python scripts/aggregate_results.py --format latex
    python scripts/aggregate_results.py --format csv
    python scripts/aggregate_results.py --format all  # outputs all 3 + json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mapping ablation_mode -> human-readable name + interpretation hint
DISPLAY_INFO = {
    "full":                ("Full Model (T+I+M+CrossAttn+Contrastive)", "Best model"),
    "text_only":           ("Text-only (PhoBERT)", "Vietnamese language signal"),
    "image_only":          ("Image-only (ViT-B/16)", "Visual features alone"),
    "metadata_only":       ("Metadata-only (17 features)", "[LEAKAGE CHECK]"),
    "text_image":          ("T + I (no metadata, no fusion)", "Cross-modal alone"),
    "text_metadata":       ("T + M (no image)", "Text + behavioral"),
    "image_metadata":      ("I + M (no text)", "Visual + behavioral"),
    "no_contrastive":      ("Full w/o contrastive loss", "Effect of InfoNCE"),
    "no_modality_dropout": ("Full w/o modality dropout", "Regularization effect"),
    "no_dropout":          ("Full w/o nn.Dropout layers", "Standard dropout regularization"),
    "no_metadata_in_fusion": ("Full w/o metadata in fusion", "Metadata fusion contribution"),
    "no_attention":        ("Full w/o cross-attention", "Effect of attention fusion"),
    "no_gating":           ("Full w/o gated fusion (sum instead)", "Effect of gating"),
}

# Display order in tables
DISPLAY_ORDER = [
    "full",
    "text_only",
    "image_only",
    "metadata_only",
    "text_image",
    "text_metadata",
    "image_metadata",
    "no_contrastive",
    "no_modality_dropout",
    "no_dropout",
    "no_metadata_in_fusion",
    "no_attention",
    "no_gating",
]


def find_results_files(results_root: Path) -> Dict[str, Dict]:
    """Find all results.json files in output directories."""
    results = {}
    
    # Walk through outputs/* directories looking for results.json
    for results_file in results_root.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
            mode = data.get("mode")
            if mode:
                results[mode] = data
                print(f"  Found: {mode} at {results_file}")
        except Exception as e:
            print(f"  Skipping {results_file}: {e}")
    
    return results


def build_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """Convert results dict to a pandas DataFrame ready for table output."""
    rows = []
    
    full_metric = None  # for delta computation
    if "full" in results and results["full"].get("status") == "success":
        full_metric = results["full"].get("test_metrics", {}).get("f1_macro")
    
    for mode in DISPLAY_ORDER:
        if mode not in results:
            print(f"  WARN: No results for '{mode}'")
            continue
        
        r = results[mode]
        if r.get("status") != "success":
            rows.append({
                "mode": mode,
                "name": DISPLAY_INFO[mode][0],
                "status": r.get("status", "unknown"),
                "f1_macro": None,
                "f1_pos": None,
                "auc_roc": None,
                "auc_pr": None,
                "accuracy": None,
                "precision_pos": None,
                "recall_pos": None,
                "delta_f1": None,
                "total_params": None,
                "trainable_params": None,
                "interpretation": "FAILED",
            })
            continue
        
        v = r.get("verification", {})
        t = r.get("test_metrics", {})
        f1 = t.get("f1_macro")
        delta = (f1 - full_metric) if (f1 is not None and full_metric is not None) else None
        
        rows.append({
            "mode": mode,
            "name": DISPLAY_INFO[mode][0],
            "status": "OK",
            "f1_macro": f1,
            "f1_pos": t.get("f1_pos"),
            "auc_roc": t.get("auc_roc"),
            "auc_pr": t.get("auc_pr"),
            "accuracy": t.get("accuracy"),
            "precision_pos": t.get("precision_pos"),
            "recall_pos": t.get("recall_pos"),
            "delta_f1": delta,
            "total_params": v.get("total_params"),
            "trainable_params": v.get("trainable_params"),
            "interpretation": DISPLAY_INFO[mode][1],
        })
    
    return pd.DataFrame(rows)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Generate a markdown table for the paper."""
    lines = []
    lines.append("## Ablation Study Results")
    lines.append("")
    lines.append("| Configuration | Total Params | F1-macro | Delta | AUC-ROC | Recall+ | Interpretation |")
    lines.append("|---|---|---|---|---|---|---|")
    
    for _, row in df.iterrows():
        if row["status"] != "OK":
            lines.append(f"| {row['name']} | - | - | - | - | - | FAILED |")
            continue
        
        params = f"{row['total_params'] / 1e6:.1f}M" if row['total_params'] else "-"
        f1 = f"{row['f1_macro']:.4f}" if pd.notnull(row['f1_macro']) else "-"
        
        if pd.notnull(row['delta_f1']):
            sign = "+" if row['delta_f1'] >= 0 else ""
            delta = f"{sign}{row['delta_f1']:.4f}"
        else:
            delta = "—"
        
        auc = f"{row['auc_roc']:.4f}" if pd.notnull(row['auc_roc']) else "-"
        recall = f"{row['recall_pos']:.4f}" if pd.notnull(row['recall_pos']) else "-"
        interp = row['interpretation']
        
        lines.append(f"| {row['name']} | {params} | {f1} | {delta} | {auc} | {recall} | {interp} |")
    
    return "\n".join(lines)


def format_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Params & F1-macro & $\Delta$ & AUC & Recall$^+$ \\")
    lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        if row["status"] != "OK":
            continue
        
        params = f"{row['total_params'] / 1e6:.1f}M" if row['total_params'] else "-"
        f1 = f"{row['f1_macro']:.4f}" if pd.notnull(row['f1_macro']) else "-"
        
        if pd.notnull(row['delta_f1']):
            sign = "+" if row['delta_f1'] >= 0 else ""
            delta = f"{sign}{row['delta_f1']:.4f}"
        else:
            delta = "—"
        
        auc = f"{row['auc_roc']:.4f}" if pd.notnull(row['auc_roc']) else "-"
        recall = f"{row['recall_pos']:.4f}" if pd.notnull(row['recall_pos']) else "-"
        
        # Escape special LaTeX chars in name
        name = row['name'].replace("&", r"\&")
        
        # Bold the full model row
        if row['mode'] == "full":
            lines.append(rf"\textbf{{{name}}} & {params} & \textbf{{{f1}}} & {delta} & {auc} & {recall} \\")
            lines.append(r"\midrule")
        else:
            lines.append(rf"{name} & {params} & {f1} & {delta} & {auc} & {recall} \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Ablation study on Vietnamese Facebook ad misinformation dataset. "
                 r"$\Delta$ is the F1-macro change vs full model.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def format_csv(df: pd.DataFrame) -> str:
    """Generate CSV output."""
    return df.to_csv(index=False)


def print_console_summary(df: pd.DataFrame):
    """Pretty-print results to stdout."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    
    for _, row in df.iterrows():
        name = row['name'][:48].ljust(48)
        if row["status"] != "OK":
            print(f"  {name}  FAILED")
            continue
        
        params = f"{row['total_params'] / 1e6:.1f}M".rjust(7)
        f1 = f"{row['f1_macro']:.4f}" if pd.notnull(row['f1_macro']) else "  -   "
        
        delta_str = ""
        if pd.notnull(row['delta_f1']):
            sign = "+" if row['delta_f1'] >= 0 else ""
            delta_str = f" (Delta={sign}{row['delta_f1']:+.4f})"
        
        auc = f"{row['auc_roc']:.4f}" if pd.notnull(row['auc_roc']) else "  -   "
        
        print(f"  {name}  {params}  F1={f1}{delta_str}  AUC={auc}")
    
    print("=" * 100)
    
    # Validation warnings
    print("\nVALIDATION CHECKS:")
    
    # Check 1: metadata_only >= 0.70 indicates leakage
    meta_row = df[df['mode'] == 'metadata_only']
    if len(meta_row) > 0 and meta_row.iloc[0]['status'] == 'OK':
        meta_f1 = meta_row.iloc[0]['f1_macro']
        if pd.notnull(meta_f1) and meta_f1 > 0.70:
            print(f"  [WARN] metadata_only F1={meta_f1:.4f} > 0.70 -- likely page-level leakage")
            print(f"         Run scripts/find_leaking_feature.py to identify the culprit")
        else:
            print(f"  [OK] metadata_only F1={meta_f1:.4f} (healthy)")
    
    # Check 2: Expected ordering
    text_row = df[df['mode'] == 'text_only']
    image_row = df[df['mode'] == 'image_only']
    if len(text_row) > 0 and len(image_row) > 0:
        text_f1 = text_row.iloc[0]['f1_macro']
        image_f1 = image_row.iloc[0]['f1_macro']
        if pd.notnull(text_f1) and pd.notnull(image_f1):
            if image_f1 > text_f1:
                print(f"  [WARN] image_only F1={image_f1:.4f} > text_only F1={text_f1:.4f}")
                print(f"         Suspicious -- text usually carries more signal in Vietnamese text data")
            else:
                print(f"  [OK] text_only ({text_f1:.4f}) > image_only ({image_f1:.4f}) (expected)")
    
    # Check 3: Full model > all ablations
    full_row = df[df['mode'] == 'full']
    if len(full_row) > 0 and full_row.iloc[0]['status'] == 'OK':
        full_f1 = full_row.iloc[0]['f1_macro']
        single_modal_max = df[df['mode'].isin(['text_only', 'image_only', 'metadata_only'])]['f1_macro'].max()
        if pd.notnull(full_f1) and pd.notnull(single_modal_max):
            improvement = full_f1 - single_modal_max
            if improvement > 0.02:
                print(f"  [OK] Full model F1 ({full_f1:.4f}) exceeds best single-modal ({single_modal_max:.4f}) "
                      f"by {improvement:.4f}")
            elif improvement > 0:
                print(f"  [WARN] Full model F1 ({full_f1:.4f}) only marginally beats best single-modal "
                      f"({single_modal_max:.4f}) -- fusion may not be helping")
            else:
                print(f"  [ERROR] Full model F1 ({full_f1:.4f}) WORSE than single-modal ({single_modal_max:.4f}) "
                      f"-- fusion is HURTING performance, investigate")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results into paper-ready tables"
    )
    parser.add_argument("--results-dir", default="outputs",
                        help="Root directory containing per-ablation output dirs")
    parser.add_argument("--format", choices=["markdown", "latex", "csv", "all"],
                        default="all")
    parser.add_argument("--output-dir", default="outputs/ablation_results",
                        help="Where to save aggregated tables")
    args = parser.parse_args()
    
    print(f"Searching for results in: {args.results_dir}")
    results = find_results_files(Path(args.results_dir))
    print(f"Found {len(results)} ablation results\n")
    
    if not results:
        print("No results found. Did you run scripts/run_ablations.py first?")
        sys.exit(1)
    
    df = build_dataframe(results)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Always print to console
    print_console_summary(df)
    
    # Save requested formats
    if args.format in ("markdown", "all"):
        md = format_markdown_table(df)
        out = output_dir / "ablation_table.md"
        out.write_text(md)
        print(f"  Markdown saved to: {out}")
    
    if args.format in ("latex", "all"):
        tex = format_latex_table(df)
        out = output_dir / "ablation_table.tex"
        out.write_text(tex)
        print(f"  LaTeX saved to: {out}")
    
    if args.format in ("csv", "all"):
        csv = format_csv(df)
        out = output_dir / "ablation_table.csv"
        out.write_text(csv)
        print(f"  CSV saved to: {out}")
    
    if args.format == "all":
        out = output_dir / "ablation_results.json"
        df.to_json(out, orient="records", indent=2)
        print(f"  JSON saved to: {out}")
    
    print()


if __name__ == "__main__":
    main()
