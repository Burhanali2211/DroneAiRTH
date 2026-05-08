"""
simulate.py — Load trained model, run mission, save plots.

Usage:
    python simulate.py
    python simulate.py --jam_start 8 --jam_dur 15
"""
import argparse
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from src.model.drone_ai import DroneAI
from src.simulator.mission_simulator import MissionSimulator
from src.visualization.plotter import plot_all


def main(jam_start: float = 6.0, jam_dur: float = 12.0):
    print("\n" + "=" * 65)
    print("DRONE AI — MISSION SIMULATION")
    print("=" * 65)

    model_dir = Path('models')
    required  = ['drone_ai_model.keras', 'scaler.json', 'feature_names.txt']
    missing   = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"[ERROR] Model files not found in models/: {missing}")
        print("        Run  python train.py  first to train and save the model.")
        sys.exit(1)

    ai  = DroneAI.load(model_dir=str(model_dir))
    sim = MissionSimulator(ai)
    sim.run(duration=30.0, dt=0.02, jam_start=jam_start, jam_dur=jam_dur)

    # Need history object for plots — load stub if training was separate
    class _HistoryStub:
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    history = ai.history if ai.history is not None else _HistoryStub()
    plot_all(history, sim, ai, output_dir='outputs')
    print("Plots saved to outputs/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jam_start', type=float, default=6.0)
    parser.add_argument('--jam_dur',   type=float, default=12.0)
    args = parser.parse_args()
    main(args.jam_start, args.jam_dur)
