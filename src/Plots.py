import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and Clean Data
df = pd.read_csv('Data/data.csv')
df.columns = df.columns.str.strip()
df = df.drop(columns=['index']).drop_duplicates().dropna()

# 2. Calculate Correlation Matrix
# We exclude T1 and P1 as they are constant and result in NaN correlations
corr_matrix = df.corr()

# 3. Plot Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Marine Vessel Propulsion Plant Parameters')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 4. Save
plt.savefig('correlation_heatmap.png')
plt.show()