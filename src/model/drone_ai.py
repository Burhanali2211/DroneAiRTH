"""
drone_ai.py — LSTM-based anti-jamming decision model with confidence hysteresis.

Architecture: (window=10, features=21) → LSTM(32) → Dense(64) → Dense(32) → softmax(4)

Key improvements over Dense-only v1:
  - LSTM sees 10 consecutive frames (0.2s at 50Hz) → temporal context
  - Confidence hysteresis: action only switches when new action wins
    majority vote over 3 consecutive predictions AND confidence > CONF_THRESHOLD
  - Flat predict() still works for backward compat (replicates frame window)
  - TFLite export uses fixed-shape signature for RPi deployment
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')

from src.config_loader import get_config


def _mcfg():
    try:
        c = get_config()
        return c.get('model', {})
    except Exception:
        return {}


# Hysteresis thresholds — read from config at call time; fall back to safe defaults
def _conf_threshold():  return _mcfg().get('conf_threshold',  0.72)
def _vote_window():     return _mcfg().get('vote_window',     3)
def _sequence_window(): return _mcfg().get('sequence_window', 10)

# Module-level aliases for backward compat (e.g. train.py passing epochs)
CONF_THRESHOLD  = 0.72
VOTE_WINDOW     = 3
SEQUENCE_WINDOW = 10


class DroneAI:
    """
    LSTM anti-jamming decision model with confidence hysteresis.

    Inputs : (SEQUENCE_WINDOW, n_features) normalised sensor sequence
    Outputs: [CONTINUE_TO_TARGET, HOVER_AT_TARGET, RETURN_HOME, EMERGENCY_LAND]

    Hysteresis prevents action thrashing on flickering signals:
    - Proposed action must appear >= VOTE_WINDOW times in a row
    - AND confidence must exceed CONF_THRESHOLD
    - Otherwise current action holds
    """

    ACTION_NAMES = [
        'CONTINUE_TO_TARGET',
        'HOVER_AT_TARGET',
        'RETURN_HOME',
        'EMERGENCY_LAND',
    ]

    def __init__(self, feature_names: list[str],
                 window: int = None):
        self.feature_names = feature_names
        self.window        = window if window is not None else _sequence_window()
        self.n_features    = len(feature_names)
        self.scaler        = StandardScaler()
        self.model         = self._build()
        self.history       = None
        self._cm           = None

        # Runtime hysteresis state (reset on each new mission)
        self._frame_buffer  = deque(maxlen=self.window)
        self._vote_buffer   = deque(maxlen=_vote_window())
        self._current_action = 0
        self._current_conf   = 0.0

    # ── Model architecture ────────────────────────────────────────────────────

    def _build(self) -> keras.Model:
        cfg = _mcfg()
        lstm_units = cfg.get('lstm_units', 32)
        dense1     = cfg.get('dense1',     64)
        dense2     = cfg.get('dense2',     32)
        lr         = cfg.get('learning_rate', 3e-4)

        inp = keras.Input(shape=(self.window, self.n_features), name='sensor_seq')

        x = layers.LSTM(lstm_units, name='lstm')(inp)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(dense1, name='d1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(dense2, name='d2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.20)(x)

        x = layers.Dense(16, activation='relu', name='d3')(x)
        out = layers.Dense(4, activation='softmax', name='action')(x)

        model = keras.Model(inp, out, name='DroneAI_LSTM')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_seq: np.ndarray, y: np.ndarray,
              epochs: int = None, batch_size: int = None) -> keras.callbacks.History:
        """
        X_seq: (N, window, features) — sequence data from data_generator
        y    : (N,) class labels
        """
        cfg        = _mcfg()
        epochs     = epochs     if epochs     is not None else cfg.get('epochs',     60)
        batch_size = batch_size if batch_size is not None else cfg.get('batch_size', 64)

        print(f"\n[MODEL] Architecture: LSTM({self.window} frames x {self.n_features} features)")
        print(f"[MODEL] Parameters:   {self.model.count_params()}")

        # Normalise per-feature across all frames
        N, W, F = X_seq.shape
        flat = X_seq.reshape(-1, F)
        self.scaler.fit(flat)
        X_norm = self.scaler.transform(flat).reshape(N, W, F).astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y)

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-5),
        ]

        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        y_pred   = np.argmax(self.model.predict(X_te, verbose=0), axis=1)
        self._cm = confusion_matrix(y_te, y_pred)
        print("\n[MODEL] Classification Report:")
        print(classification_report(y_te, y_pred, target_names=self.ACTION_NAMES))

        return self.history

    # ── Inference with hysteresis ─────────────────────────────────────────────

    def reset_state(self):
        """Call at the start of each mission to clear hysteresis state."""
        self._frame_buffer.clear()
        self._vote_buffer.clear()
        self._current_action = 0
        self._current_conf   = 0.0

    def predict(self, sensor_list: list) -> tuple[int, float, np.ndarray]:
        """
        Feed one 21-feature frame. Internally maintains a rolling window.
        Returns (action_id, confidence, all_probs).

        Hysteresis: the returned action_id only changes when VOTE_WINDOW
        consecutive raw predictions agree AND confidence > CONF_THRESHOLD.
        This prevents thrashing on flickering GPS/RC signals.
        """
        # Normalise single frame
        frame = np.array(sensor_list, dtype=np.float32)
        frame_norm = self.scaler.transform(frame.reshape(1, -1))[0]
        self._frame_buffer.append(frame_norm)

        # Pad with first frame if buffer not full yet
        if len(self._frame_buffer) < self.window:
            seq = np.array([self._frame_buffer[0]] *
                           (self.window - len(self._frame_buffer)) +
                           list(self._frame_buffer))
        else:
            seq = np.array(self._frame_buffer)

        # Raw model prediction
        probs  = self.model.predict(seq[np.newaxis], verbose=0)[0]
        raw_action = int(np.argmax(probs))
        raw_conf   = float(probs[raw_action])

        # EMERGENCY_LAND bypasses hysteresis — always immediate
        if raw_action == 3 and raw_conf > 0.60:
            self._current_action = 3
            self._current_conf   = raw_conf
            self._vote_buffer.clear()
            return 3, raw_conf, probs

        # Accumulate votes
        self._vote_buffer.append(raw_action)

        # Switch only if majority vote agrees + confidence high enough
        if (len(self._vote_buffer) >= _vote_window() and
                all(v == raw_action for v in self._vote_buffer) and
                raw_conf >= _conf_threshold()):
            self._current_action = raw_action
            self._current_conf   = raw_conf

        return self._current_action, self._current_conf, probs

    def predict_sequence(self, seq: np.ndarray) -> tuple[int, float, np.ndarray]:
        """
        Direct sequence prediction — no hysteresis.
        seq: (window, features) raw unscaled
        """
        N, F   = seq.shape
        flat   = self.scaler.transform(seq.reshape(-1, F)).reshape(1, N, F)
        probs  = self.model.predict(flat.astype(np.float32), verbose=0)[0]
        action = int(np.argmax(probs))
        return action, float(probs[action]), probs

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: str = 'models'):
        out = Path(model_dir)
        out.mkdir(exist_ok=True)

        keras_path = out / 'drone_ai_model.keras'
        self.model.save(str(keras_path))
        print(f"[MODEL] Saved: {keras_path}")

        scaler_data = {
            'mean':  self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'window': self.window,
            'n_features': self.n_features,
        }
        (out / 'scaler.json').write_text(json.dumps(scaler_data, indent=2))
        (out / 'feature_names.txt').write_text('\n'.join(self.feature_names))
        (out / 'window_size.txt').write_text(str(self.window))
        print(f"[MODEL] Scaler + metadata saved to {model_dir}/")

    def to_tflite(self, model_dir: str = 'models'):
        """Int8-quantized TFLite export — smallest size for RPi 3."""
        out  = Path(model_dir)
        path = out / 'drone_ai_model.tflite'

        # Representative dataset for full-integer quantization
        N, F = 200, self.n_features
        W    = self.window
        dummy = np.random.randn(N, W, F).astype(np.float32)

        def rep_data():
            for i in range(N):
                yield [dummy[i:i+1]]

        conv = tf.lite.TFLiteConverter.from_keras_model(self.model)
        conv.optimizations          = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_data
        try:
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type      = tf.int8
            conv.inference_output_type     = tf.int8
            data = conv.convert()
            print("[MODEL] TFLite: int8 quantization")
        except Exception:
            # Fallback to float16 if int8 fails
            conv2 = tf.lite.TFLiteConverter.from_keras_model(self.model)
            conv2.optimizations = [tf.lite.Optimize.DEFAULT]
            conv2.target_spec.supported_types = [tf.float16]
            data = conv2.convert()
            print("[MODEL] TFLite: float16 quantization (int8 fallback)")

        path.write_bytes(data)
        kb = path.stat().st_size / 1024
        print(f"[MODEL] TFLite saved: {path} ({kb:.1f} KB)")

    @classmethod
    def load(cls, model_dir: str = 'models') -> 'DroneAI':
        d = Path(model_dir)
        keras_path  = d / 'drone_ai_model.keras'
        scaler_path = d / 'scaler.json'
        feat_path   = d / 'feature_names.txt'

        missing = [str(p) for p in (keras_path, scaler_path, feat_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[MODEL] Missing model files in '{model_dir}/': {missing}. "
                "Run train.py first to generate them."
            )

        try:
            feature_names = feat_path.read_text().splitlines()

            window = _sequence_window()
            if (d / 'window_size.txt').exists():
                window = int((d / 'window_size.txt').read_text().strip())

            instance = cls(feature_names, window=window)
            instance.model = keras.models.load_model(str(keras_path))

            sd = json.loads(scaler_path.read_text())
            instance.scaler.mean_          = np.array(sd['mean'])
            instance.scaler.scale_         = np.array(sd['scale'])
            instance.scaler.n_features_in_ = len(feature_names)

        except Exception as e:
            raise RuntimeError(
                f"[MODEL] Failed to load model from '{model_dir}/': {e}"
            ) from e

        print(f"[MODEL] Loaded LSTM model from {model_dir}/ (window={window})")
        return instance
