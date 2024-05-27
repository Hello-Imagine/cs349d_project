def dnn(X_train,y_train):
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
  from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
  # Create a Sequential model
  model = Sequential()
  input_dim = X_train.shape[1]
  print(input_dim)
  model.add(Dense(256, input_dim=input_dim, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  #model.add(Dense(128, input_dim=input_dim, activation='relu'))
  # Input layer and first hidden layer with 64 units and ReLU activation
  model.add(Dense(64, input_dim=input_dim, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  # Second hidden layer with 32 units and ReLU activation
  model.add(Dense(32, activation='relu'))
  # Summary of the model
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  # Output layer with units equal to the number of classes and softmax activation
  model.add(Dense(1, activation='softmax'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  # Compile the model

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]
  model.summary()
  history = model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks = callbacks)
  model_filepath = "classifier_dnn.pth"
  with open(model_filepath, 'wb+') as f:
    pickle.dump(model, f)
  return model