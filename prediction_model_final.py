import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Load data
data = pd.read_csv('subscriber_data.csv')
features = data.drop(['subscriber_id', 'subscription_status', 'cancellationRequested'], axis=1)
target = data['cancellationRequested']

# Prepare data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply random oversampling to balance the data
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(scaled_features, target)

# Split the balanced data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Save the model
model.save('balanced_model.h5')

# Make predictions
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy:", accuracy)

# Create a DataFrame with UUID and churn prediction percentage
uuids = data['subscriber_id']
churn_percentage_predictions = model.predict(scaled_features) * 100  # Scale to percentage

result_df = pd.DataFrame({
    'subscriber_id': uuids,
    'churn-prediction_percentage': churn_percentage_predictions.flatten()
})

# Write the result DataFrame to a CSV file
result_df.to_csv('balanced_churn_predictions.csv', index=False)
print("Churn predictions saved to balanced_churn_predictions.csv")

# Display the DataFrame
print(result_df)

print("Execution completed.")
