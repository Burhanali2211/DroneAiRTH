"""
main.py — Full pipeline: generate data → train → simulate → plot.

Usage:
    python main.py
    python main.py --samples 20000 --epochs 80 --jam_start 8 --jam_dur 15
"""
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure Unicode output works on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.model.data_generator import generate_training_data, load_cached
from src.model.drone_ai import DroneAI
from src.simulator.mission_simulator import MissionSimulator
from src.visualization.plotter import plot_all


def main(samples=10000, epochs=60, jam_start=6.0, jam_dur=12.0, use_cache=False):
    print("\n" + "=" * 65)
    print("ANTI-JAMMING DRONE AI — FULL PIPELINE")
    print("=" * 65)

    # 1. Data
    print("\n[STEP 1] Training data")
    if use_cache:
        cached = load_cached('data')
        if cached:
            X_seq, X_flat, y, feature_names = cached
        else:
            X_seq, X_flat, y, feature_names = generate_training_data(samples, cache_dir='data')
    else:
        X_seq, X_flat, y, feature_names = generate_training_data(samples, cache_dir='data')

    # 2. Model
    print("\n[STEP 2] Train model")
    ai = DroneAI(feature_names)
    ai.train(X_seq, y, epochs=epochs)

    # 3. Save
    print("\n[STEP 3] Save models")
    ai.save(model_dir='models')
    ai.to_tflite(model_dir='models')

    # 4. Simulate
    print("\n[STEP 4] Mission simulation")
    sim = MissionSimulator(ai)
    sim.run(duration=30.0, dt=0.02, jam_start=jam_start, jam_dur=jam_dur)

    # 5. Visualise
    print("\n[STEP 5] Generate plots")
    plot_all(ai.history, sim, ai, output_dir='outputs')

    # 6. Summary
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Parameters:   {ai.model.count_params()}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  Train acc:    {ai.history.history['accuracy'][-1]:.2%}")
    print(f"  Val acc:      {ai.history.history['val_accuracy'][-1]:.2%}")
    print(f"  Mission:      {'SUCCESS' if sim.success else 'INCOMPLETE'}")
    print("\n  outputs/flight_simulation.png")
    print("  outputs/training_history.png")
    print("  models/drone_ai_model.keras")
    print("  models/drone_ai_model.tflite   <- for Raspberry Pi 3")
    print("  models/scaler.json")
    print("  data/X.npy  data/y.npy")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',   type=int,   default=10000)
    parser.add_argument('--epochs',    type=int,   default=60)
    parser.add_argument('--jam_start', type=float, default=6.0)
    parser.add_argument('--jam_dur',   type=float, default=12.0)
    parser.add_argument('--use_cache', action='store_true',
                        help='Reuse cached training data from data/')
    args = parser.parse_args()
    main(args.samples, args.epochs, args.jam_start, args.jam_dur, args.use_cache)
