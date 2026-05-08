"""
train.py — Generate data, train model, save everything to disk.

Usage:
    python train.py
    python train.py --samples 20000 --epochs 80
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from src.model.data_generator import generate_training_data
from src.model.drone_ai import DroneAI


def main(samples: int = 10000, epochs: int = 60):
    print("\n" + "=" * 65)
    print("DRONE AI — TRAINING")
    print("=" * 65)

    X_seq, X_flat, y, feature_names = generate_training_data(num_samples=samples, cache_dir='data')

    ai = DroneAI(feature_names)
    ai.train(X_seq, y, epochs=epochs)

    ai.save(model_dir='models')
    ai.to_tflite(model_dir='models')

    final_acc  = ai.history.history['accuracy'][-1]
    final_vacc = ai.history.history['val_accuracy'][-1]
    print(f"\n  Train acc:  {final_acc:.2%}")
    print(f"  Val acc:    {final_vacc:.2%}")
    print("  Models saved to models/")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs',  type=int, default=60)
    args = parser.parse_args()
    main(args.samples, args.epochs)
