# 2. Define Features and Target
# Using the Compressor decay coefficient as our primary target
target = 'GT Compressor decay state coefficient'
X = df.drop(columns=['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient'])
y = df[target]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42), # Built-in alternative
    "SVM (SVR)": SVR(kernel='rbf', C=1.0, epsilon=0.01)
}

# 6. Evaluation Loop
results = {}
plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(models.items()):
    # Use scaled data for all for consistency
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {"MAE": mae, "R2": r2}

    # Visualization: Actual vs Predicted Plot
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test, predictions, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{name}\nR2: {r2:.4f}')
    plt.xlabel('Actual Coefficient')
    plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# 7. Print Performance Comparison
print("\nModel Performance Comparison:")
comparison_df = pd.DataFrame(results).T
print(comparison_df)
