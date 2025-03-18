# -*- coding: utf-8 -*-
"""Assignment1_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WqyaFERkv0wuF7iw59fqb9E83vTArjYH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""a. Select any 5 features out of the 10 provided below. Implement forward stepwise linear
regression with the chosen features. The process involves iteratively adding one feature at
a time from your selection. After adding each feature, evaluate the model's performance
using metrics such as Mean Squared Error, R-squared, Adjusted R-squared, or BIC-Bayesian
Information Criterion (preferred). Choose the feature that contributes the most to
improving the model's performance and add it to the model. Continue this iterative
process for a total of 5 iterations. Explain your selection criteria for adding or removing
features.
"""

df=pd.read_csv('Cancer_dataset.csv')

df.head(10)

columns_to_select=['mean_radius',
'mean_perimeter',
'worst_symmetry',
'lymph_node_status',
'mean_texture',]

df[columns_to_select]=df[columns_to_select].fillna(0)

print(df[columns_to_select].head(10))

X=df[columns_to_select].to_numpy()
print("Original X_train: \n",X[0:10])
print(len(X))

Y=df[['tumor_size']].fillna(0)
Y=np.array(Y)
print("Original Y_train: \n",Y[:10])
print(len(Y))
print(X.shape)
print(Y.shape)

#feature scaling
X_train_copy=X.copy()
Y_train_copy=Y.copy()

n=len(X)
indices=np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
split_indices=int(0.8*n)
X_train=X[indices[ :split_indices]]
X_test=X[indices[split_indices: ]]
Y_train=Y[indices[ :split_indices]]
Y_test=Y[indices[split_indices: ]]

#Function for BIC

def calculate_BIC(n, k, y, res_ss):
  L=(-n/2)*np.log(2*np.pi)-(n/2)*np.log(res_ss/n)-(res_ss/(2*np.var(y)))
  BIC=np.log(n)*k-2*L
  return BIC

#function for AIC
def calculate_aic(n, k, y, res_ss):
  L=(-n/2)*np.log(2*np.pi)-(n/2)*np.log(res_ss/n)-(res_ss/(2*np.var(y)))

  AIC=2*k-2*L
  return AIC

n = len(Y_train)  # Number of training samples
k = 1  # Only the intercept is used
SSE_bias = np.sum((Y_train - Y_train.mean()) ** 2)  # Sum of Squares for bias model

bias_Bic = calculate_BIC(n, k, Y_train, SSE_bias)  # Compute BIC
print(f"BIC for Bias-Only Model: {bias_Bic}")

bias_Aic= calculate_aic( n, k, Y_train, SSE_bias)
print(f"AIC for Bias-Only Model: {bias_Aic}")

#function to fit ols model
def ols(x, y):
    # Add a column of ones to X for the intercept term (beta0)
    X_ = np.c_[np.ones(len(x)), x]

    # Compute the pseudo-inverse of X^T * X
    x_inv = np.linalg.pinv(X_.T @ X_)#@ is nothing but the dot product(np.dot) made simpler

    # Compute the regression coefficients (beta)
    beta = x_inv @ (X_.T) @ y

    return beta

def MSE(predy, y):
  mse=np.mean((predy-y)**2)
  return mse

def forward_step():
  selected_columns = []  # Start with an empty list of selected features
  remaining_features = list(range(X_train.shape[1]))  # List of all feature indices
  current_bic = bias_Bic  # Start with the bias-only model's BIC

  # Stepwise regression
  for i in range(5):  # upto to 5 iterations
      best_bic = current_bic
      best_feature = None
      best_model = None
      best_beta = None

      # Try adding each remaining feature
      for feature in remaining_features:
          new_features = selected_columns + [feature]  # Add current feature to selected features
          model = X_train[:, new_features]  # Select the columns for the current model
          beta = ols(model, Y_train)  # Fit the OLS model

          # Make predictions and calculate residual sum of squares (RSS)
          y_pred = np.c_[np.ones(model.shape[0]), model] @ beta
          rss = np.sum((Y_train - y_pred) ** 2)  # Residual sum of squares

          # Calculate BIC for this model
          bic = calculate_BIC(n, len(new_features) + 1, Y_train, rss)  # Use the number of features
          # Keep the best feature based on BIC
          if bic < best_bic:
              best_bic = bic
              best_feature = feature
              best_model = model
              best_beta = beta

      # If BIC improves, add the feature and update remaining features
      if best_feature is not None:
          selected_columns.append(best_feature)  # Add the best feature to selected
          remaining_features.remove(best_feature)  # Remove the best feature from remaining
          current_bic = best_bic  # Update the current BIC
          print(f"Iteration {i+1}: Added feature {best_feature}, BIC: {best_bic}")
      else:
          print(f"Bic value obtained: {bic} which is larger than the best bic obtained before so the loop stoped")
          break
  return selected_columns, best_bic

selected_columns, bestBic = forward_step()
print(len(selected_columns))
# Final selected features
for i in range(len(selected_columns)):
  Best_features=columns_to_select[selected_columns[i]]
  print(Best_features)
print(f"Best bic: {bestBic}")

print(selected_columns)

X_train_selected = X_train[:, selected_columns]
X_test_selected = X_test[:, selected_columns]

# Fit the model on the training set
beta = ols(X_train_selected, Y_train)

# Make predictions on the test set
X_test_with_intercept = np.c_[np.ones(X_test_selected.shape[0]), X_test_selected]  # Add intercept term
y_pred = X_test_with_intercept @ beta  # Predictions on test set

# Calculate the Mean Squared Error (MSE) aldready have a function called MSE
mse =MSE(y_pred, Y_test)  # Calculate MSE on test set
print(f"Mean Squared error for forward stepwise regression: {mse}")

def r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_rs = np.sum((y_true - y_pred) ** 2)
    ss_tss=np.sum((y_true-mean_y)**2)
    r2 = 1 - (ss_rs / ss_tss)
    return r2

r_sq=r_squared(Y_test,y_pred)
print(f"R_Squared for forward stepwise regression: {r_sq}")

# Calculate Adjusted R-squared
n = len(Y_test)  # Number of data points in the test set
p = X_test_selected.shape[1]  # Number of predictors (features) in the selected model
r_squared_adj = 1 - ((1 - r_sq) * (n - 1)) / (n - p - 1)


print(f"Adjusted R-squared for forward stepwise regression: {r_squared_adj}")

"""**Backward stepwise regression**
The goal here is that by removing predictors with lower impact (less meaningful)and the model becomes simpler without losing predictive power, and this simplification leads to a lower BIC.
"""

colums_for_back=['mean_radius',
'mean_symmetry',
'mean_perimeter',
'mean_fractal_dimension',
'worst_symmetry',
'lymph_node_status',
'mean_area',
'mean_smoothness',
'worst_radius',
'worst_area', ]

df[colums_for_back]=df[colums_for_back].fillna(0)
print(df[colums_for_back].head(10))

X=df[colums_for_back].to_numpy()
print(X[:10])

n=len(X)
print(n)
indices=np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
split=int(0.8*n)
X_train=X[indices[ :split]]
X_test=X[indices[split: ]]
print(f"X_train before standardization: \n{X_train[:10]}")
print(f"X_test before standardization: \n{X_test[:10]}")

selected_columns=list(range(len(colums_for_back)))
print(selected_columns)

def back_stepwise(X_train, Y_train, n, selected_columns):


    # Start with all predictors and calculate the BIC for the full model
    beta = ols(X_train, Y_train)
    y_predict = np.c_[np.ones(X_train.shape[0]), X_train] @ beta
    k = len(selected_columns) + 1
    res_s = np.sum((Y_train - y_predict) ** 2)
    full_bic = calculate_BIC(n, k, Y_train, res_s)
    print("Initial full model BIC:", full_bic)

    trac_features_removed = []

    for i in range(5):  # Limit to 5 iterations or stop if no features can be removed
        print(f"Iteration {i+1}:")
        bic_values = {}

        # Evaluate removing each feature one at a time
        for feature in selected_columns:
            print(f"Evaluating feature: {feature}")
            reduced_features = []
            for f in selected_columns:
              if f!=feature:
                reduced_features.append(f)

            if not reduced_features:  # Prevents empty feature selection
                continue

            X_reduced = X_train[:, reduced_features]
            beta_reduced = ols(X_reduced, Y_train)
            y_predict_reduced = np.c_[np.ones(len(Y_train)), X_reduced] @ beta_reduced
            k_reduced = len(reduced_features) + 1
            res_s_reduced = np.sum((Y_train - y_predict_reduced) ** 2)
            bic_values[feature] = calculate_BIC(n, k_reduced, Y_train, res_s_reduced)
            print(f"BIC for feature {feature}: {bic_values[feature]}")
        # Stop if no valid feature removal
        if not bic_values:
            print("No more features can be removed.")
            break

        # Identify the feature with the lowest BIC and remove it
        min_bic = min(bic_values, key=bic_values.get)
        print(f"Removing feature {min_bic} with BIC: {bic_values[min_bic]}")

        selected_columns.remove(min_bic)
        trac_features_removed.append(min_bic)

        # Recompute the full model BIC after feature removal
        X_new = X_train[:, selected_columns]
        beta_new = ols(X_new, Y_train)
        y_predict_new = np.c_[np.ones(len(Y_train)), X_new] @ beta_new
        k_new = len(selected_columns) + 1
        res_s_new = np.sum((Y_train - y_predict_new) ** 2)
        full_bic = calculate_BIC(n, k_new, Y_train, res_s_new)

    print("Final selected columns:", selected_columns)
    print("Number of selected predictors:", len(selected_columns))
    print(f"Features removed: {trac_features_removed}")
    print(f"Full bic: {full_bic}")
    return selected_columns, full_bic

n = len(Y_train)  # Number of samples in the training set
selected_columns,bic = back_stepwise(X_train, Y_train, n, selected_columns)

print("The features obtained are: ")
for i in range(len(selected_columns)):
  choosen_features=selected_columns[i]
  print(colums_for_back[choosen_features] ,end =", ")

X_train_selected_back = X_train[:, selected_columns]
X_test_selected_back = X_test[:, selected_columns]

# Fit the model on the training set (X_train_selected)
beta = ols(X_train_selected_back, Y_train)

# Make predictions on the test set (X_test_selected)
y_pred = np.c_[np.ones(X_test_selected_back.shape[0]), X_test_selected_back] @ beta  # Predictions on test set

# Calculate the Mean Squared Error (MSE)
mse_back =MSE(y_pred, Y_test)  # Calculate MSE on test set
print(f"Mean Squared error: {mse_back}")

rsq=r_squared(Y_test, y_pred)
print(f"R_Squared: {rsq}")

# Calculate Adjusted R-squared
n = len(Y_test)  # Number of data points in the test set
p = X_test_selected.shape[1]  # Number of predictors (features) in the selected model
rsquared_adj = 1 - ((1 - rsq) * (n - 1)) / (n - p - 1)


print(f"Adjusted R-squared: {rsquared_adj}")

"""Compare the final model obtained from forward stepwise regression with the final model
obtained from baciward stepwise regression. Which one is better? Discuss the differences
in terms of the selected features, model performance
"""

#compare both forward and backward

# Data
metrics = ["MSE", "R²", "Adjusted R²"]
Forward_stepwise = [mse, r_sq, r_squared_adj]
Backward_stepwise = [mse_back, rsq, rsquared_adj]

# Convert to DataFrame for Seaborn
data = pd.DataFrame({
    "Metric": metrics,
    "Forward Stepwise": Forward_stepwise,
    "Backward Stepwise": Backward_stepwise
})

# Melt for Seaborn compatibility
data_melted = data.melt(id_vars="Metric", var_name="Model", value_name="Value")

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Value", hue="Model", data=data_melted, palette=["blue", "red"])
plt.title("Comparison of Forward Stepwise vs Backward Stepwise")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.show()

#comparing bic
metrics=['BIC']
model_1=[bestBic]
model_2=[bic]
data=pd.DataFrame({
    "Metric": metrics,
    "Forward Stepwise Regression": model_1,
    "Backward Stepwise Regression": model_2
})

data_melted=data.melt(id_vars="Metric", var_name="Model", value_name="Value")
plt.figure(figsize=(8,5))
sns.barplot(x="Metric", y="Value", hue="Model", data=data_melted, palette=["blue", "red"])
plt.title("Comparison of Forward regression vs Backward regression")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

# Performance Comparison Table
# Model                   |	MSE |  	R²	  | Adjusted R² |
#                         ------------------------------|
# Q.2 Multiple Regression	|17.12|	 0.013 	|  -0.042     |
# Forward Stepwise	      | 0.8 |	 0.270	|   0.230     |
# Backward Stepwise      	| 0.6 |	 0.276	|   0.237     |

# the values for multipple regression are obtained from the Q(2) rounded to 3 digits
mse_r=17.12022366806357

# Data
metrics = ["MSE"]
model_1 = [mse]
model_2 = [mse_back]
model_3 = [mse_r]

# Convert to DataFrame for Seaborn
data = pd.DataFrame({
    "Metric": metrics,
    "Forward regression": model_1,
    "Backward regression": model_2,
    "Multiple regression": model_3
})

# Melt for Seaborn compatibility
data_melted = data.melt(id_vars="Metric", var_name="Model", value_name="Value")

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Value", hue="Model", data=data_melted, palette=["blue", "red", "gold"])
plt.title("Comparison of Forward regression vs Backward regression vs Multiple regression")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

rsq_r=0.0113
rsq_adj_r=-0.042

#Data
metrics=['R²','Adjusted R²']
model_1=[r_sq, r_squared_adj]
model_2=[rsq, rsquared_adj]
model_3=[rsq_r, rsq_adj_r]

data=pd.DataFrame({
    "Metric": metrics,
    "Forward Stepwise Regression": model_1,
    "Backward Stepwise Regression": model_2,
    "Multiple Regression": model_3
})

data_melted=data.melt(id_vars="Metric", var_name="Model", value_name="Value")

#Plot
plt.figure(figsize=(8,5))
sns.barplot(x="Metric", y="Value", hue="Model", data=data_melted, palette=["blue", "red", "gold"])
plt.title("Comparison of Forward regression vs Backward regression vs Multiple regression")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

"""d.

"""

#comparing bic
metrics=['BIC']
model_1=[bestBic]
model_2=[592.94]
data=pd.DataFrame({
    "Metric": metrics,
    "Forward Stepwise Regression": model_1,
    "Multiple Regression": model_2
})

data_melted=data.melt(id_vars="Metric", var_name="Model", value_name="Value")
plt.figure(figsize=(8,5))
sns.barplot(x="Metric", y="Value", hue="Model", data=data_melted, palette=["blue", "red"])
plt.title("Comparison of Multiple regression vs Backward regression")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()