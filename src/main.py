import matplotlib.pyplot as plt
import os
from CleaningData import load_and_clean_data, split_and_save_data


def run_analysis():
    # 1. Load and clean the main data
    df = load_and_clean_data()

    # 2. Create the training (30%) and testing (70%) CSV files
    # This will save train.csv and test.csv into your Data/ folder
    train_df, test_df = split_and_save_data(df)

    # 3. Define path for the images folder
    # This ensures images are saved at the same level as src and Data
    if os.path.basename(os.getcwd()) == 'src':
        image_dir = os.path.join('..', 'images')
    else:
        image_dir = 'images'

    # Create the directory if it does not exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"Created directory: {image_dir}")

    # Column definitions for plotting
    gt_torque_col = 'Gas Turbine (GT) shaft torque (GTT) [kN m]'
    fuel_col = 'Fuel flow (mf) [kg/s]'
    speed_col = 'Ship speed (v)'
    prop_torques = ['Starboard Propeller Torque (Ts) [kN]', 'Port Propeller Torque (Tp) [kN]']

    unique_speeds = sorted(df[speed_col].unique())
    plt.close('all')

    for speed in unique_speeds:
        # Filter data for the specific speed
        speed_df = df[df[speed_col] == speed].reset_index(drop=True)

        # --- Plot 1: GT Torque & Propeller Torques ---
        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes1[0].plot(speed_df[gt_torque_col], color='red')
        axes1[0].set_title(f'Gas Turbine (GT) Shaft Torque at Speed {speed}')
        axes1[0].set_ylabel('Torque [kN m]')
        axes1[0].grid(True)

        axes1[1].plot(speed_df[prop_torques[0]], color='blue', label='Starboard')
        axes1[1].plot(speed_df[prop_torques[1]], color='green', linestyle='--', label='Port')
        axes1[1].set_title(f'Propeller Torques at Speed {speed}')
        axes1[1].set_ylabel('Torque [kN]')
        axes1[1].legend()
        axes1[1].grid(True)
        plt.tight_layout()

        # Save Plot 1 to the images folder
        fig1_filename = os.path.join(image_dir, f'propeller_torque_speed_{speed}.png')
        plt.savefig(fig1_filename)
        plt.show()
        plt.close(fig1)

        # --- Plot 2: Turbine Torque & Fuel Flow ---
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(speed_df[gt_torque_col], color='tab:red', linewidth=1.5)
        ax1.set_title(f'Turbine Torque at Speed {speed}')
        ax1.set_ylabel('Torque [kN m]')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(speed_df[fuel_col], color='tab:orange', linewidth=1.5)
        ax2.set_title(f'Fuel Flow Rate at Speed {speed}')
        ax2.set_ylabel('Fuel flow [kg/s]')
        ax2.set_xlabel('Observation Index')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save Plot 2 to the images folder
        fig2_filename = os.path.join(image_dir, f'torque_and_fuel_speed_{speed}.png')
        plt.savefig(fig2_filename)
        print(f"Saved: {fig2_filename}")
        plt.show()
        plt.close(fig2)


if __name__ == "__main__":
    run_analysis()