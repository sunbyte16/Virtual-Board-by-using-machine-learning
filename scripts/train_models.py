"""
Script to train ML models for Virtual Board
"""
import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml_models import DigitRecognizer, LetterRecognizer
from src.config import config

def train_digit_model(epochs=None, save_model=True):
    """Train digit recognition model"""
    print("="*50)
    print("TRAINING DIGIT RECOGNITION MODEL")
    print("="*50)
    
    recognizer = DigitRecognizer()
    
    # Override epochs if specified
    if epochs:
        config.ml.epochs = epochs
    
    print(f"Training for {config.ml.epochs} epochs...")
    
    start_time = datetime.now()
    history = recognizer.train_on_mnist(save_model=save_model)
    end_time = datetime.now()
    
    training_time = end_time - start_time
    print(f"\nTraining completed in: {training_time}")
    
    # Print training summary
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"Final training accuracy: {final_accuracy:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    
    return recognizer

def create_sample_letter_data():
    """Create sample letter training data (placeholder)"""
    print("="*50)
    print("CREATING SAMPLE LETTER DATA")
    print("="*50)
    
    import numpy as np
    from tensorflow import keras
    
    # This is a placeholder - in a real implementation, you would
    # load actual letter dataset like EMNIST
    print("Note: This creates dummy data for demonstration.")
    print("For real letter recognition, use EMNIST dataset.")
    
    # Create dummy data (26 classes for A-Z)
    num_samples = 1000
    x_data = np.random.rand(num_samples, 28, 28, 1)
    y_data = np.random.randint(0, 26, num_samples)
    
    return x_data, y_data

def train_letter_model(epochs=None, save_model=True):
    """Train letter recognition model"""
    print("="*50)
    print("TRAINING LETTER RECOGNITION MODEL")
    print("="*50)
    
    recognizer = LetterRecognizer()
    
    # Create model
    model = recognizer.create_model()
    
    # Get sample data (in real implementation, use EMNIST)
    x_data, y_data = create_sample_letter_data()
    
    # Split data
    split_idx = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    # Train model
    epochs = epochs or 3  # Fewer epochs for dummy data
    
    print(f"Training for {epochs} epochs...")
    
    start_time = datetime.now()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    end_time = datetime.now()
    
    training_time = end_time - start_time
    print(f"\nTraining completed in: {training_time}")
    
    if save_model:
        recognizer.model = model
        recognizer.save_model()
    
    return recognizer

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train ML models for Virtual Board')
    parser.add_argument('--model', choices=['digit', 'letter', 'both'], default='digit',
                       help='Which model to train (default: digit)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save trained models')
    
    args = parser.parse_args()
    
    save_models = not args.no_save
    
    print("VIRTUAL BOARD - MODEL TRAINING")
    print("="*60)
    print(f"Training model(s): {args.model}")
    print(f"Save models: {save_models}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    print("="*60)
    
    try:
        if args.model in ['digit', 'both']:
            digit_model = train_digit_model(epochs=args.epochs, save_model=save_models)
            print("✓ Digit model training completed")
        
        if args.model in ['letter', 'both']:
            letter_model = train_letter_model(epochs=args.epochs, save_model=save_models)
            print("✓ Letter model training completed")
        
        print("\n" + "="*60)
        print("ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if save_models:
            print("Models saved to 'models/' directory")
        
        print("\nYou can now run the Virtual Board:")
        print("  python -m src.main")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()