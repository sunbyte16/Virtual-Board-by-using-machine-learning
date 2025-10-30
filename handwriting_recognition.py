import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class HandwritingRecognizer:
    def __init__(self):
        self.model = None
        self.input_shape = (28, 28, 1)
        
    def create_model(self):
        """Create a CNN model for digit recognition"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
        
    def train_on_mnist(self):
        """Train the model on MNIST dataset"""
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
            
        # Train the model
        history = self.model.fit(x_train, y_train,
                               epochs=5,
                               batch_size=128,
                               validation_data=(x_test, y_test),
                               verbose=1)
        
        return history
        
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Reshape for model
        image = image.reshape(1, 28, 28, 1)
        
        return image
        
    def predict_digit(self, image):
        """Predict digit from image"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
        
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)