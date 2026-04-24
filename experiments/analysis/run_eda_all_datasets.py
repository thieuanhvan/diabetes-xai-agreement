"""
Run Exploratory Data Analysis (EDA) for all registered datasets.

All generated outputs will be saved to:
    <PROJECT_ROOT>/outputs/eda/<dataset_name>/
"""

from utils.project_paths import OUTPUT_DIR
from datasets.dataset_registry import DATASETS
from dataset_profiler import load_dataset
from dataset_visualizer import visualize_dataset


EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"


def run_eda(dataset_name: str):
    print("\n" + "=" * 70)
    print(f"Running EDA for dataset: {dataset_name}")
    print("=" * 70)

    X, y, cfg = load_dataset(dataset_name)

    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")
    print("Generating plots...")

    dataset_output = EDA_OUTPUT_DIR / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    visualize_dataset(
        dataset_name=dataset_name,
        X=X,
        y=y,
        dataset_type=cfg.get("task", "classification"),
        output_dir=dataset_output,
    )

    print(f"EDA completed for {dataset_name}")
    print("Saved to:", dataset_output.resolve())


def main():
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS.keys():
        run_eda(dataset_name)


if __name__ == "__main__":
    main()