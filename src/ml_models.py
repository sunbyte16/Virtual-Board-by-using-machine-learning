"""
Machine Learning models for Virtual Board
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from typing import Tuple, Optional, List
from src.config import config
from src.utils import preprocess_for_ml

class DigitRecognizer:
    """CNN model for digit recognition"""
    
    def __init__(self):
        self.model = None
        self.input_shape = config.ml.input_shape
        self.model_path = config.ml.model_path
        
    def create_model(self) -> keras.Model:
        """Create CNN model architecture"""
        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            
            # Dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_on_mnist(self, save_model: bool = True) -> keras.callbacks.History:
        """Train model on MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(x_train)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        print("Training model...")
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=config.ml.batch_size),
            epochs=config.ml.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        if save_model:
            self.save_model()
        
        return history
    
    def predict_digit(self, image: np.ndarray) -> Tuple[int, float]:
        """Predict digit from image"""
        if self.model is None:
            raise ValueError("Model not loaded! Train or load a model first.")
        
        # Preprocess image
        processed_image = preprocess_for_ml(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Predict digits for batch of images"""
        if self.model is None:
            raise ValueError("Model not loaded! Train or load a model first.")
        
        # Preprocess all images
        processed_images = np.vstack([preprocess_for_ml(img) for img in images])
        
        # Make predictions
        predictions = self.model.predict(processed_images, verbose=0)
        
        results = []
        for pred in predictions:
            digit = np.argmax(pred)
            confidence = np.max(pred)
            results.append((digit, confidence))
        
        return results
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        if filepath is None:
            filepath = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> None:
        """Load pre-trained model"""
        if filepath is None:
            filepath = self.model_path
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

class LetterRecognizer:
    """CNN model for letter recognition (A-Z)"""
    
    def __init__(self):
        self.model = None
        self.input_shape = config.ml.input_shape
        self.model_path = "models/letter_model.h5"
        self.num_classes = 26  # A-Z
    
    def create_model(self) -> keras.Model:
        """Create CNN model for letter recognition"""
        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            
            # Dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def predict_letter(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict letter from image"""
        if self.model is None:
            raise ValueError("Model not loaded! Train or load a model first.")
        
        # Preprocess image
        processed_image = preprocess_for_ml(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Convert to letter
        predicted_letter = chr(ord('A') + predicted_class)
        
        return predicted_letter, confidence
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        if filepath is None:
            filepath = self.model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Letter model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> None:
        """Load pre-trained model"""
        if filepath is None:
            filepath = self.model_path
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Letter model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Letter model file not found: {filepath}")

class ModelManager:
    """Manager for all ML models"""
    
    def __init__(self):
        self.digit_recognizer = DigitRecognizer()
        self.letter_recognizer = LetterRecognizer()
        self.current_mode = "digit"  # "digit" or "letter"
    
    def setup_models(self) -> None:
        """Setup and load/train models"""
        # Setup digit recognizer
        if os.path.exists(self.digit_recognizer.model_path):
            print("Loading pre-trained digit model...")
            self.digit_recognizer.load_model()
        else:
            print("Training new digit model...")
            self.digit_recognizer.train_on_mnist()
        
        # Setup letter recognizer (optional)
        if os.path.exists(self.letter_recognizer.model_path):
            print("Loading pre-trained letter model...")
            self.letter_recognizer.load_model()
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict based on current mode"""
        if self.current_mode == "digit":
            digit, confidence = self.digit_recognizer.predict_digit(image)
            return str(digit), confidence
        elif self.current_mode == "letter":
            if self.letter_recognizer.model is not None:
                return self.letter_recognizer.predict_letter(image)
            else:
                return "N/A", 0.0
        else:
            return "Unknown", 0.0
    
    def switch_mode(self, mode: str) -> None:
        """Switch between digit and letter recognition"""
        if mode in ["digit", "letter"]:
            self.current_mode = mode
            print(f"Switched to {mode} recognition mode")