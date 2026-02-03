import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from CleaningData import load_and_clean_data, split_and_save_data
from Plots import run_all_plots

# 1. Setup paths to include the models folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(project_root, 'models')
if models_path not in sys.path:
    sys.path.append(models_path)

# 2. Imports from your local modules
# Ensure these filenames match your .py files exactly (e.g., Gradientboosting.py)
from Random_forest import train_random_forest
from Gradientboosting import train_gradient_boosting
from SVM import train_svm


def main():
    # --- 1. Prepare Data ---
    df = load_and_clean_data()
    split_and_save_data(df)

    # --- 2. Define Paths ---
    if os.path.basename(os.getcwd()) == 'src':
        train_path = os.path.join('..', 'Data', 'train.csv')
        image_dir = os.path.join('..', 'images')
    else:
        train_path = os.path.join('Data', 'train.csv')
        image_dir = 'images'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # --- 3. Run Plotting ---
    print("Starting plotting process...")
    run_all_plots(df, image_dir)
    print("Plotting tasks completed successfully.")

    # --- 4. Run Machine Learning Models (Training Data Only) ---
    print("\n--- Training Models on 30% Training Set ---")

    # Passing the newly defined train_path and image_dir to your functions
    rf_results = train_random_forest(train_path, image_dir)
    gb_results = train_gradient_boosting(train_path, image_dir)
    svm_results = train_svm(train_path, image_dir)

    # --- 5. Performance Summary ---
    results_df = pd.DataFrame([rf_results, gb_results, svm_results])
    print("\nTraining Performance Summary:")
    print(results_df)


if __name__ == "__main__":
    main()