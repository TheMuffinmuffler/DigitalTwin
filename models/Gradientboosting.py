import pandas as pd
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_gradient_boosting(train_path, image_dir):
    train_df = pd.read_csv(train_path)

    target = 'GT Compressor decay state coefficient'
    drop_cols = ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)
    mae = mean_absolute_error(y_train, predictions)
    r2 = r2_score(y_train, predictions)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, predictions, alpha=0.5, s=10, color='orange')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.title(f'XGBoost (Training Set)\nR2: {r2:.4f} | MAE: {mae:.4f}')
    plt.xlabel('Actual Coefficient')
    plt.ylabel('Predicted')

    save_path = os.path.join(image_dir, 'ml_xgboost_training_only.png')
    plt.savefig(save_path)
    plt.close()

    return {"Model": "XGBoost", "MAE": mae, "R2": r2}