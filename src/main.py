import matplotlib.pyplot as plt
import os
from CleaningData import load_and_clean_data, split_and_save_data


import os
from CleaningData import load_and_clean_data, split_and_save_data
from Plots import run_all_plots

def main():
    # 1. Prepare Data
    # Clean the raw data and create the 30/70 train-test split files
    df = load_and_clean_data()
    split_and_save_data(df)

    # 2. Setup Images Directory
    # Ensure images are saved to a folder at the project root level
    if os.path.basename(os.getcwd()) == 'src':
        image_dir = os.path.join('..', 'images')
    else:
        image_dir = 'images'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 3. Run Plotting
    # Call the Plotting module to generate the heatmap and speed graphs
    print("Starting plotting process...")
    run_all_plots(df, image_dir)
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()