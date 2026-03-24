Algorithm:
Step 1: Start
Step 2: Import Required Libraries
  Import NumPy, Pandas
  Import TensorFlow/Keras
  Import dataset and preprocessing libraries
Step 3: Load Dataset
  Load the dataset
  Store input features in X
  Store target labels in y
Step 4: Split Dataset
  Divide dataset into:
  Training set (80%)
  Testing set (20%)
Step 5: Preprocess Data
  Apply standardization using scaler:
  Fit scaler on training data
  Transform both training and testing data
Step 6: Initialize ANN Model
  Create a Sequential model
Step 7: Add Layers to ANN
  Add input layer with number of features
  Add hidden layers with activation function (ReLU)
  Add output layer with Softmax activation
Step 8: Compile Model
  Select optimizer (Adam)
  Select loss function (categorical/sparse categorical crossentropy)
  Define evaluation metric (accuracy)
Step 9: Train the Model
  Fit model on training data:
  Set number of epochs
  Set batch size
  Use validation split
Step 10: Evaluate Model
  Test the model using test dataset
Compute:
  Test accuracy
  Test loss
Step 11: Display Results
  Print accuracy and loss
Step 12: Stop
