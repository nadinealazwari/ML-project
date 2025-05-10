import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

# Load dataset
data = pd.read_csv('Original data\hour.csv')
print(f"\nDataset Shape: {data.shape}")
print(f"Number of Attributes: {data.shape[1]}")
print(f"Number of Samples: {data.shape[0]}")

# Data Properties and Statistics
print("\n=== Data Statistics ===")
print(data.describe())

print("\n=== Data Types ===")
print(data.dtypes)

print("\n=== Missing Values ===")
print("Missing values per column:")
print(data.isnull().sum())
print("\nNo missing values found in this dataset.")

# 3. Data Preprocessing
print("\n=== Preprocessing Steps ===")
print("1. Removed 'instant' (record index) - Non-predictive")
print("2. Kept 'dteday' but converted to datetime features")
print("3. Standardized all numerical features (except target)")
print("   - Reason: Different scales would bias distance-based algorithms")
print("4. Converted categorical variables to proper data types")
print("5. Created temporal features from date")

# Preprocessing implementation
data = data.drop(['instant'], axis=1)
data['dteday'] = pd.to_datetime(data['dteday'])
data['dayofweek'] = data['dteday'].dt.dayofweek
data['is_weekend'] = data['dayofweek'].isin([5,6]).astype(int)

# Convert categoricals
categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
data[categorical_cols] = data[categorical_cols].astype('category')

# Separate features and target
X = data.drop(['cnt', 'casual', 'registered', 'dteday'], axis=1)
y = data['cnt']


# 4. Train-Test Split
print("\n=== Train-Test Split ===")
print("Split ratio: 80% training, 20% testing")
print("Random state fixed for reproducibility")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
numerical_cols = ['temp', 'atemp', 'hum', 'windspeed', 'dayofweek']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save preprocessed data
pd.DataFrame(X_train, columns=X.columns).to_csv('Preprocessed data/X.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('Preprocessed data/X_test.csv', index=False)
y_train.to_csv('Preprocessed data/Y.csv', index=False)
y_test.to_csv('Preprocessed data/Y_test.csv', index=False)


# 5. Advanced Visualizations of data
plt.figure(figsize=(20, 15))

# Plot 1: Temporal Patterns
plt.subplot(3, 2, 1)
sns.boxplot(x='hr', y='cnt', data=data)
plt.title('Hourly Rental Patterns')

# Plot 2: Weather Impact
plt.subplot(3, 2, 2)
sns.barplot(x='weathersit', y='cnt', data=data)
plt.title('Rentals by Weather Condition')

# Plot 3: Correlation Heatmap
plt.subplot(3, 2, 3)
corr = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')

# Plot 4: Daily Trends
plt.subplot(3, 2, 4)
daily = data.groupby('dteday')['cnt'].sum().reset_index()
sns.lineplot(x='dteday', y='cnt', data=daily)
plt.title('Daily Rental Trends')

# Plot 5: Distribution Analysis
plt.subplot(3, 2, 5)
sns.histplot(y, bins=50, kde=True)
plt.title('Target Variable Distribution')

# Plot 6: Seasonality
plt.subplot(3, 2, 6)
sns.boxplot(x='season', y='cnt', data=data)
plt.title('Seasonal Rental Patterns')

plt.tight_layout()
plt.savefig('advanced_visualizations.png')
plt.show()


# 6. Model Training and Evaluation
models = {
    'SVM': SVR(),
    'Random_Forest': RandomForestRegressor(),
    'Decision_Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
    'Naive_Bayes': BayesianRidge(),
    'ANN': MLPRegressor(max_iter=5000)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

     # Save predictions
    pd.DataFrame({'Actual': y_test, 'Predicted': preds}).to_csv(f'Results/prediction_{name}_model.csv', index=False)
    
    results.append({
        'Model': name,
        'R2': r2_score(y_test, preds),
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds))
    })

results_df = pd.DataFrame(results)
print("\n=== Model Performance ===")
print(results_df.sort_values('R2', ascending=False))

# Save results
results_df.to_csv('model_performance.csv', index=False)

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: R2 Scores
plt.subplot(2, 2, 1)
sns.barplot(x='R2', y='Model', data=results_df.sort_values('R2', ascending=False))
plt.title('Model Comparison: R² Scores')
plt.xlim(0, 1)

# Plot 2: MAE
plt.subplot(2, 2, 2)
sns.barplot(x='MAE', y='Model', data=results_df.sort_values('MAE'))
plt.title('Model Comparison: MAE (Lower is Better)')

# Plot 3: RMSE
plt.subplot(2, 2, 3)
sns.barplot(x='RMSE', y='Model', data=results_df.sort_values('RMSE'))
plt.title('Model Comparison: RMSE (Lower is Better)')

# Plot 4: Actual vs Predicted for best model
best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
best_preds = pd.read_csv(f'Results/prediction_{best_model_name}_model.csv')

plt.subplot(2, 2, 4)
plt.scatter(best_preds['Actual'], best_preds['Predicted'], alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Best Model ({best_model_name}): Actual vs Predicted')

plt.tight_layout()
plt.savefig('Results/model_performance_comparison.png')
plt.show()

# 7. Advanced Visualization of Model Results
plt.figure(figsize=(20, 15))

# Plot 1: Model Performance Comparison
plt.subplot(2, 2, 1)
sns.set_style("whitegrid")
performance_melt = results_df.melt(id_vars='Model', 
                                  value_vars=['R2', 'MAE', 'RMSE'],
                                  var_name='Metric', 
                                  value_name='Score')
sns.barplot(x='Score', y='Model', hue='Metric', 
            data=performance_melt.sort_values('Score', ascending=False),
            palette='viridis')
plt.title('Model Performance Comparison', fontsize=14)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Actual vs Predicted for Top 3 Models
top_models = results_df.nlargest(3, 'R2')['Model'].values
plt.subplot(2, 2, 2)
for model_name in top_models:
    preds = pd.read_csv(f'Results/prediction_{model_name}_model.csv')['Predicted']
    sns.regplot(x=y_test, y=preds, 
                label=f'{model_name} (R²={results_df[results_df.Model==model_name].R2.values[0]:.3f})',
                scatter_kws={'alpha':0.3})
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
plt.xlabel('Actual Count', fontsize=12)
plt.ylabel('Predicted Count', fontsize=12)
plt.title('Actual vs Predicted (Top 3 Models)', fontsize=14)
plt.legend()

# Plot 3: Error Distribution Comparison
plt.subplot(2, 2, 3)
for model_name in top_models:
    preds = pd.read_csv(f'Results/prediction_{model_name}_model.csv')['Predicted']
    errors = y_test - preds
    sns.kdeplot(errors, label=model_name, bw_adjust=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Error Distribution Comparison', fontsize=14)
plt.legend()

# Plot 4: Feature Importance for Best Model
plt.subplot(2, 2, 4)
best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
if hasattr(models[best_model_name], 'feature_importances_'):
    importances = models[best_model_name].feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
    # sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket')
    sns.barplot(x='Importance', y='Feature', data=importance_df, 
                hue='Feature', palette='rocket', dodge=False, legend=False)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14)
elif hasattr(models[best_model_name], 'coef_'):
    coefs = models[best_model_name].coef_
    features = X_train.columns
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
    coef_df = coef_df.sort_values('Coefficient', ascending=False).head(10)
    # sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='rocket')
    sns.barplot(x='Coefficient', y='Feature', data=coef_df,
                hue='Feature', palette='rocket', dodge=False, legend=False)
    plt.title(f'Feature Coefficients - {best_model_name}', fontsize=14)
else:
    plt.text(0.5, 0.5, f'No feature importance\navailable for {best_model_name}',
             ha='center', va='center', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14)

plt.tight_layout()
plt.savefig('Results/model_results_visualization.png', dpi=300, bbox_inches='tight')
plt.show()