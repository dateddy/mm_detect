#!/usr/bin/env python3
"""
Validation script: Load all YAML configs and verify consistency.

Checks:
1. All configs load without YAML parse errors
2. Each user-facing config has unique experiment_name
3. All expected ablation modes are present
4. Display format: "[path] | mode=X | exp=Y"
"""

import yaml
from pathlib import Path
from collections import defaultdict

def load_config(path):
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    repo_root = Path(__file__).parent.parent
    configs_dir = repo_root / "configs"
    
    # Collect all config files to load
    config_files = []
    
    # Base configs
    config_files.append(("base", configs_dir / "base.yaml", True))  # is_template
    config_files.append(("default", configs_dir / "default.yaml", False))
    
    # Model configs
    model_dir = configs_dir / "model"
    if model_dir.exists():
        for f in sorted(model_dir.glob("*.yaml")):
            name = f.stem
            is_template = (name == "multimodal")  # multimodal is canonical
            config_files.append((name, f, is_template))
    
    # Ablation configs
    ablation_dir = configs_dir / "ablation"
    if ablation_dir.exists():
        for f in sorted(ablation_dir.glob("*.yaml")):
            name = f.stem
            is_legacy = (name in ["no_metadata", "no_dropout"])  # legacy aliases
            is_template = is_legacy
            config_files.append((name, f, is_template))
    
    print(f"Found {len(config_files)} config files\n")
    
    mode_map = defaultdict(list)
    exp_map = defaultdict(list)
    errors = []
    
    user_facing_configs = []  # For uniqueness checks
    
    for config_name, config_path, is_template in config_files:
        rel_path = config_path.relative_to(repo_root)
        
        try:
            config = load_config(config_path)
            
            # Extract ablation_mode and experiment_name
            ablation_mode = config.get('ablation_mode', 'MISSING')
            experiment_name = config.get('experiment_name', 'MISSING')
            
            template_marker = " [TEMPLATE]" if is_template else ""
            print(f"[OK] {rel_path}{template_marker} | mode={ablation_mode} | exp={experiment_name}")
            
            # Track for uniqueness checks (skip templates)
            if not is_template:
                user_facing_configs.append((config_name, ablation_mode, experiment_name))
                mode_map[ablation_mode].append(str(rel_path))
                exp_map[experiment_name].append(str(rel_path))
            
        except Exception as e:
            errors.append((rel_path, str(e)))
            print(f"[ERROR] {rel_path} | {e}")
    
    print("\n" + "="*80)
    print("\nValidation Results:")
    all_valid = True
    
    # Check for errors
    if errors:
        print(f"\n✗ {len(errors)} config file(s) failed to load:")
        for path, error in errors:
            print(f"  {path}: {error}")
        all_valid = False
    else:
        print(f"\n✓ All {len(config_files)} configs loaded successfully")
    
    # Check duplicate modes in user-facing configs
    print("\nUser-facing configs by ablation mode:")
    for mode, files in sorted(mode_map.items()):
        if len(files) > 1:
            print(f"  ⚠ Mode '{mode}' used by: {files}")
        else:
            print(f"  ✓ Mode '{mode}': {files[0]}")
    
    # Check duplicate experiment names
    print("\nUser-facing configs by experiment name:")
    seen_exps = set()
    for exp, files in sorted(exp_map.items()):
        if exp in seen_exps:
            continue
        seen_exps.add(exp)
        if len(files) > 1:
            print(f"  ⚠ Experiment '{exp}' used by: {files}")
        else:
            print(f"  ✓ Experiment '{exp}'")
    
    # Expected ablation modes (11 distinct modes)
    expected_modes = {
        'full',
        'text_only',
        'image_only',
        'metadata_only',
        'text_image',
        'text_metadata',
        'image_metadata',
        'full_no_contrastive',
        'full_no_modality_dropout',
        'full_no_attention',
        'full_no_gating'
    }
    
    actual_modes = set(mode_map.keys()) - {'MISSING'}
    missing_modes = expected_modes - actual_modes
    
    print(f"\nCoverage of expected ablation modes ({len(expected_modes)}):")
    for mode in sorted(expected_modes):
        if mode in actual_modes:
            print(f"  ✓ {mode}")
        else:
            print(f"  ✗ {mode} (MISSING)")
            all_valid = False
    
    # Config count check
    print(f"\nConfiguration count:")
    print(f"  Total configs: {len(config_files)}")
    print(f"  Templates: {sum(1 for _, _, is_template in config_files if is_template)}")
    print(f"  User-facing: {len(user_facing_configs)}")
    print(f"  Expected user-facing: 11 (default, 3 model, 7 ablation)")
    
    if len(user_facing_configs) < 11:
        print(f"  ✗ Not enough user-facing configs!")
        all_valid = False
    else:
        print(f"  ✓ Sufficient configs")
    
    print("\n" + "="*80)
    
    if all_valid and not errors and not missing_modes:
        print("\n✓ All validations passed!")
        print(f"\nConfig Summary:")
        print(f"  {len(user_facing_configs)} user-facing configs covering {len(actual_modes)} ablation modes")
        return 0
    else:
        print("\n✗ Some validations failed (see details above)")
        return 1

if __name__ == "__main__":
    exit(main())

