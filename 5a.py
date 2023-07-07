import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate sample data
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1)

# Define KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize list to store evaluation metrics
eval_metrics = []

# Iterate through each fold
for train_index, test_index in kfold.split(X):
   # Split data into training and testing sets
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   # Define and compile model
   model = Sequential()
   model.add(Dense(64, activation='relu', input_dim=10))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   # Fit model to training data
   model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

   # Evaluate model on testing data
   eval_metrics.append(model.evaluate(X_test, y_test))

# Print average evaluation metrics across all folds
print("Average evaluation metrics:")
print("Loss:", np.mean([m[0] for m in eval_metrics]))
print("MAE:", np.mean([m[1] for m in eval_metrics]))
