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


# Hysteresis thresholds
CONF_THRESHOLD  = 0.72   # min confidence to accept an action switch
VOTE_WINDOW     = 3      # consecutive agreeing predictions before switching
SEQUENCE_WINDOW = 10     # frames fed to LSTM (10 × 20ms = 200ms context)


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
                 window: int = SEQUENCE_WINDOW):
        self.feature_names = feature_names
        self.window        = window
        self.n_features    = len(feature_names)
        self.scaler        = StandardScaler()
        self.model         = self._build()
        self.history       = None
        self._cm           = None

        # Runtime hysteresis state (reset on each new mission)
        self._frame_buffer  = deque(maxlen=window)   # raw unscaled frames
        self._vote_buffer   = deque(maxlen=VOTE_WINDOW)
        self._current_action = 0
        self._current_conf   = 0.0

    # ── Model architecture ────────────────────────────────────────────────────

    def _build(self) -> keras.Model:
        inp = keras.Input(shape=(self.window, self.n_features), name='sensor_seq')

        # LSTM with return_sequences=False — outputs last hidden state
        x = layers.LSTM(32, name='lstm')(inp)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, name='d1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(32, name='d2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.20)(x)

        x = layers.Dense(16, activation='relu', name='d3')(x)
        out = layers.Dense(4, activation='softmax', name='action')(x)

        model = keras.Model(inp, out, name='DroneAI_LSTM')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_seq: np.ndarray, y: np.ndarray,
              epochs: int = 60, batch_size: int = 64) -> keras.callbacks.History:
        """
        X_seq: (N, window, features) — sequence data from data_generator
        y    : (N,) class labels
        """
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
        if (len(self._vote_buffer) >= VOTE_WINDOW and
                all(v == raw_action for v in self._vote_buffer) and
                raw_conf >= CONF_THRESHOLD):
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
        feature_names = (d / 'feature_names.txt').read_text().splitlines()

        window = SEQUENCE_WINDOW
        if (d / 'window_size.txt').exists():
            window = int((d / 'window_size.txt').read_text().strip())

        instance = cls(feature_names, window=window)
        instance.model = keras.models.load_model(str(d / 'drone_ai_model.keras'))

        sd = json.loads((d / 'scaler.json').read_text())
        instance.scaler.mean_           = np.array(sd['mean'])
        instance.scaler.scale_          = np.array(sd['scale'])
        instance.scaler.n_features_in_  = len(feature_names)

        print(f"[MODEL] Loaded LSTM model from {model_dir}/ (window={window})")
        return instance
