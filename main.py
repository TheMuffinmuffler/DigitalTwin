import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('Data/data.csv')

# 2. Data Cleaning & Consistency
df.columns = df.columns.str.strip() # Remove hidden spaces
df = df.drop_duplicates()
df = df.dropna()

speed_counts = df['Ship speed (v)'].value_counts().sort_index()

print(speed_counts)

# Define the columns we want to track
gt_torque = 'Gas Turbine (GT) shaft torque (GTT) [kN m]'
prop_torques = ['Starboard Propeller Torque (Ts) [kN]', 'Port Propeller Torque (Tp) [kN]']

# Get a sorted list of all unique speeds (3, 6, 9, etc.)
unique_speeds = sorted(df['Ship speed (v)'].unique())

# Loop through each speed and generate a plot
for speed in unique_speeds:
    # Filter data for the current speed
    speed_df = df[df['Ship speed (v)'] == speed].reset_index(drop=True)

    # Create the subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Graph: GT Shaft Torque
    axes[0].plot(speed_df[gt_torque], color='red')
    axes[0].set_title(f'Gas Turbine (GT) Shaft Torque at Speed {speed}')
    axes[0].set_ylabel('Torque [kN m]')
    axes[0].grid(True)

    # Bottom Graph: Propeller Torques
    axes[1].plot(speed_df[prop_torques[0]], color='blue', label='Starboard')
    axes[1].plot(speed_df[prop_torques[1]], color='green', linestyle='--', label='Port')
    axes[1].set_title(f'Propeller Torques at Speed {speed}')
    axes[1].set_ylabel('Torque [kN]')
    axes[1].set_xlabel('Observation Index')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Save the file with the speed in the name
    plt.savefig(f'torque_at_speed_{speed}.png')

    # Close the plot to free up memory before moving to the next one
    plt.show()