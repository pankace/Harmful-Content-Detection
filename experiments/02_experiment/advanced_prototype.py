import numpy as np
import tensorflow as tf
from keras.layers import Conv1D, BatchNormalization, Dropout, MaxPooling1D, LSTM, Bidirectional, Dense, Flatten
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = tf.keras.models.Sequential()

num_samples = 1000
input_shape = (1000, 1)

X = np.random.rand(num_samples, *input_shape)
y = np.random.randint(2, size=num_samples)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

regularization_standard = 0.005
# Convulution layers
conv1d_layer_sync_dropout_min = 0.20
conv1d_layer_sync_dropout_max = 0.24
# Dense layers
dense_layer_sync_dropout_min = 0.20
dense_layer_sync_dropout_max = 0.25
#Recurring layers
recurring_layer_sync_dropout_min = 0.32
recurring_layer_sync_dropout_max = 0.36

model.add(Conv1D(filters=12, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(np.random.uniform(conv1d_layer_sync_dropout_min, conv1d_layer_sync_dropout_max)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(np.random.uniform(conv1d_layer_sync_dropout_min, conv1d_layer_sync_dropout_max)))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(20, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_standard)))
model.add(Dropout(np.random.uniform(recurring_layer_sync_dropout_min, recurring_layer_sync_dropout_max)))

model.add(Bidirectional(LSTM(24, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_standard))))
model.add(Dropout(np.random.uniform(recurring_layer_sync_dropout_min, recurring_layer_sync_dropout_max)))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_standard)))
model.add(BatchNormalization())
model.add(Dropout(np.random.uniform(dense_layer_sync_dropout_min, dense_layer_sync_dropout_max)))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
