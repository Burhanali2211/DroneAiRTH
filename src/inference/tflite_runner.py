"""
tflite_runner.py - TFLite inference with same interface as DroneAI.predict().
Loads drone_ai_model.tflite + scaler.json, maintains rolling frame buffer,
applies same hysteresis as DroneAI.
"""
import json
import numpy as np
from pathlib import Path
from collections import deque

CONF_THRESHOLD = 0.72
VOTE_WINDOW    = 3
ACTION_NAMES   = ['CONTINUE_TO_TARGET','HOVER_AT_TARGET','RETURN_HOME','EMERGENCY_LAND']

class TFLiteRunner:
    def __init__(self, model_dir='models'):
        d = Path(model_dir)
        # Load tflite interpreter
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self.interpreter = Interpreter(model_path=str(d / 'drone_ai_model.tflite'))
        self.interpreter.allocate_tensors()
        self._in  = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

        sd = json.loads((d / 'scaler.json').read_text())
        self.mean_   = np.array(sd['mean'],  dtype=np.float32)
        self.scale_  = np.array(sd['scale'], dtype=np.float32)
        self.window  = sd['window']
        self.n_features = sd['n_features']

        self.feature_names = (d / 'feature_names.txt').read_text().splitlines()

        self._frame_buffer   = deque(maxlen=self.window)
        self._vote_buffer    = deque(maxlen=VOTE_WINDOW)
        self._current_action = 0
        self._current_conf   = 0.0

    def reset_state(self):
        self._frame_buffer.clear()
        self._vote_buffer.clear()
        self._current_action = 0
        self._current_conf   = 0.0

    def predict(self, sensor_list: list) -> tuple:
        """Same signature as DroneAI.predict(). Returns (action_id, confidence, probs)."""
        frame = np.array(sensor_list, dtype=np.float32)
        frame_norm = (frame - self.mean_) / (self.scale_ + 1e-8)
        self._frame_buffer.append(frame_norm)

        if len(self._frame_buffer) < self.window:
            seq = np.array([self._frame_buffer[0]] *
                           (self.window - len(self._frame_buffer)) +
                           list(self._frame_buffer))
        else:
            seq = np.array(self._frame_buffer)

        inp = seq[np.newaxis].astype(self._in['dtype'])

        # Handle int8 quantization
        if self._in['dtype'] == np.int8:
            scale, zp = self._in['quantization']
            inp = (inp / scale + zp).astype(np.int8)

        self.interpreter.set_tensor(self._in['index'], inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self._out['index'])[0]

        # Dequantize int8 output
        if self._out['dtype'] == np.int8:
            scale, zp = self._out['quantization']
            raw = (raw.astype(np.float32) - zp) * scale

        probs      = raw.astype(np.float32)
        raw_action = int(np.argmax(probs))
        raw_conf   = float(probs[raw_action])

        if raw_action == 3 and raw_conf > 0.60:
            self._current_action = 3
            self._current_conf   = raw_conf
            self._vote_buffer.clear()
            return 3, raw_conf, probs

        self._vote_buffer.append(raw_action)
        if (len(self._vote_buffer) >= VOTE_WINDOW and
                all(v == raw_action for v in self._vote_buffer) and
                raw_conf >= CONF_THRESHOLD):
            self._current_action = raw_action
            self._current_conf   = raw_conf

        return self._current_action, self._current_conf, probs
