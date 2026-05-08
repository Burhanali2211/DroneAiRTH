"""
run_realtime.py - Real-time flight brain. Runs at 50 Hz on RPi or laptop with SITL.

Usage:
  python run_realtime.py                          # simulation mode
  python run_realtime.py --mode tcp --host 127.0.0.1 --port 5760   # SITL
  python run_realtime.py --mode serial --port /dev/ttyAMA0          # real RPi
  python run_realtime.py --tflite                 # use TFLite instead of Keras
"""
import argparse, time, csv, sys, signal
from pathlib import Path
from datetime import datetime

# must be first
import warnings; warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   default='simulation', choices=['simulation','tcp','udp','serial'])
    parser.add_argument('--host',   default='127.0.0.1')
    parser.add_argument('--port',   default='/dev/ttyAMA0')
    parser.add_argument('--mav_port', type=int, default=5760)
    parser.add_argument('--baud',   type=int, default=57600)
    parser.add_argument('--tflite', action='store_true', help='Use TFLite model (for RPi)')
    parser.add_argument('--log',    default='logs', help='Directory to write flight log CSV')
    parser.add_argument('--hz',     type=float, default=50.0)
    args = parser.parse_args()

    from src.config_loader import get_config
    cfg = get_config()

    # Load model
    if args.tflite:
        from src.inference.tflite_runner import TFLiteRunner
        ai = TFLiteRunner(model_dir='models')
        print('[BRAIN] Using TFLite model')
    else:
        from src.model.drone_ai import DroneAI
        ai = DroneAI.load(model_dir='models')
        print('[BRAIN] Using Keras model')

    # Connect sensor bridge
    from src.control.mavlink_bridge import MAVLinkBridge
    bridge = MAVLinkBridge(
        mode=args.mode, port=args.port, host=args.host,
        baud=args.baud, mav_port=args.mav_port,
    )
    bridge.connect()

    # Control layer
    from src.control.actuation     import ActionExecutor
    from src.control.state_machine import FlightStateMachine
    from src.control.watchdog      import Watchdog

    target = cfg['mission']['target']
    import numpy as np
    executor = ActionExecutor(bridge._conn, target=np.array(target),
                              cruise_alt=cfg['mission']['cruise_alt'])
    fsm      = FlightStateMachine()
    watchdog = Watchdog(bridge._conn, fsm)

    ai.reset_state()
    fsm.arm()
    watchdog.start()

    # CSV logger
    log_dir = Path(args.log)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = log_path.open('w', newline='')
    writer   = None   # init after first row so we know column names

    dt = 1.0 / args.hz
    loop_count = 0

    ACTION_NAMES = ['CONTINUE_TO_TARGET','HOVER_AT_TARGET','RETURN_HOME','EMERGENCY_LAND']

    def _shutdown(sig, frame):
        print('\n[BRAIN] Shutting down...')
        watchdog.stop()
        bridge.close()
        log_file.close()
        print(f'[BRAIN] Log saved: {log_path}')
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f'[BRAIN] Loop running at {args.hz} Hz. Ctrl-C to stop.')
    print('='*60)

    while True:
        t0 = time.monotonic()

        sensors = bridge.get_sensor_dict()
        action_id, conf, probs = ai.predict(list(sensors.values()))

        # Watchdog can override AI decision
        override = watchdog.check(sensors)
        if override is not None:
            action_id = override

        # Update FSM, execute
        fsm.apply(action_id)
        executor.execute(action_id, sensors)

        # Log
        row = {
            'time':       time.time(),
            'loop':       loop_count,
            'action_id':  action_id,
            'action':     ACTION_NAMES[action_id],
            'confidence': round(conf, 4),
            'phase':      fsm.phase.name,
            **{k: round(float(v), 4) for k, v in sensors.items()},
        }
        if writer is None:
            writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)

        # Print at 1 Hz
        if loop_count % int(args.hz) == 0:
            print(f"[t={loop_count/args.hz:6.1f}s] {ACTION_NAMES[action_id]:22} "
                  f"conf={conf:.1%} bat={sensors['battery']:.1f}% "
                  f"jam={sensors['jam_noise']:.2f} phase={fsm.phase.name}")

        loop_count += 1
        elapsed = time.monotonic() - t0
        sleep_t = max(0.0, dt - elapsed)
        if elapsed > dt * 1.5 and loop_count > 10:
            print(f"[WARN] Loop over-budget: {elapsed*1000:.1f}ms > {dt*1000:.0f}ms budget")
        time.sleep(sleep_t)

if __name__ == '__main__':
    main()
