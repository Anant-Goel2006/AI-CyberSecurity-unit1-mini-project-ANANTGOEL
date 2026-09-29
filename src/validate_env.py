"""
Environment and Workspace Validation Script
Unit 1 Capstone Project: AI in Cyber Security
"""

import sys
import os
from pathlib import Path

def validate():
    print("=" * 60)
    print("  WORKSPACE BOOTSTRAP & ENVIRONMENT SETUP VALIDATION")
    print("=" * 60)

    # 1. Python & Environment
    print(f"[✓] Python Executable : {sys.executable}")
    print(f"[✓] Python Version    : {sys.version.split()[0]}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"[✓] In Virtual Env    : {in_venv} ({sys.prefix})")

    # 2. Package Imports and Versions
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "jupyter": "jupyter",
        "ipykernel": "ipykernel"
    }

    print("\nPackage Verification:")
    all_packages_ok = True
    for display_name, module_name in packages.items():
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [✓] {display_name:<15} : {ver}")
        except ImportError as e:
            print(f"  [✗] {display_name:<15} : NOT FOUND ({e})")
            all_packages_ok = False

    # 3. Directory Structure Verification
    project_root = Path(__file__).resolve().parent.parent
    expected_folders = ["data", "src"]
    expected_files = ["README.md", ".gitignore", "requirements.txt"]
    expected_datasets = [
        project_root / "data" / "UNSW_NB15_training-set.csv",
        project_root / "data" / "UNSW_NB15_testing-set.csv"
    ]

    print("\nDirectory & File Verification:")
    for folder in expected_folders:
        folder_path = project_root / folder
        status = "[✓]" if folder_path.is_dir() else "[✗]"
        print(f"  {status} Directory: {folder}/")

    for file_name in expected_files:
        file_path = project_root / file_name
        status = "[✓]" if file_path.is_file() else "[✗]"
        print(f"  {status} File     : {file_name}")

    print("\nDataset Verification:")
    datasets_ok = True
    for ds_path in expected_datasets:
        if ds_path.is_file():
            size_mb = ds_path.stat().st_size / (1024 * 1024)
            print(f"  [✓] Found: {ds_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  [✗] Missing: {ds_path.name}")
            datasets_ok = False

    print("\n" + "=" * 60)
    if all_packages_ok and datasets_ok:
        print("  STATUS: ALL SYSTEM CHECKS PASSED SUCCESSFULLY!")
    else:
        print("  STATUS: WARNING - SOME CHECKS FAILED.")
    print("=" * 60)

if __name__ == "__main__":
    validate()
