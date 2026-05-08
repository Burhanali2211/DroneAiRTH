from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend: always saves, never blocks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COLOR_JAM  = '#ff4444'
COLOR_NORM = '#4488ff'


def _shade_jam(ax, t: np.ndarray, jam_mask: np.ndarray):
    """Shade jamming windows on a time-series axis."""
    in_jam = False
    t_start = 0.0
    for i, jm in enumerate(jam_mask):
        if jm and not in_jam:
            t_start = t[i]; in_jam = True
        elif not jm and in_jam:
            ax.axvspan(t_start, t[i], alpha=0.15, color='red')
            in_jam = False
    if in_jam:
        ax.axvspan(t_start, t[-1], alpha=0.15, color='red')


def plot_all(history, simulator, ai, output_dir: str = 'outputs'):
    """
    Render 12-panel flight-analysis figure + standalone training-history figure.
    Both are saved to output_dir/ before plt.show() so the GUI never blocks saving.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    has_history = bool(history.history.get('loss'))

    df       = pd.DataFrame(simulator.log)
    jam_mask = df['jammed'].values.astype(bool)
    t        = df['t'].values

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Anti-Jamming Drone AI — Flight Analysis', fontsize=14, fontweight='bold')

    # ── Row 1 ─────────────────────────────────────────────────────────

    # 1A  Loss
    ax1 = fig.add_subplot(3, 4, 1)
    if has_history:
        ax1.plot(history.history['loss'],     label='Train', color='steelblue')
        ax1.plot(history.history['val_loss'], label='Val',   color='orange')
    else:
        ax1.text(0.5, 0.5, 'No training history\n(model loaded from disk)',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=9)
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # 1B  Accuracy
    ax2 = fig.add_subplot(3, 4, 2)
    if has_history:
        ax2.plot(history.history['accuracy'],     label='Train', color='steelblue')
        ax2.plot(history.history['val_accuracy'], label='Val',   color='orange')
    else:
        ax2.text(0.5, 0.5, 'No training history\n(model loaded from disk)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=9)
    ax2.set_title('Training Accuracy'); ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # 1C  Confusion matrix
    ax3 = fig.add_subplot(3, 4, 3)
    if ai._cm is not None:
        cm = ai._cm
        ax3.imshow(cm, cmap='Blues')
        short = ['CONT', 'HOVER', 'RTH', 'LAND']
        ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
        ax3.set_xticklabels(short, fontsize=8)
        ax3.set_yticklabels(short, fontsize=8)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
        for i in range(4):
            for j in range(4):
                ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=9,
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # 1D  2D trajectory
    ax4 = fig.add_subplot(3, 4, 4)
    x_pos = df['pos_x'].values
    y_pos = df['pos_y'].values
    for i in range(1, len(df)):
        c = COLOR_JAM if jam_mask[i] else COLOR_NORM
        ax4.plot(x_pos[i-1:i+1], y_pos[i-1:i+1], color=c, linewidth=2, alpha=0.8)
    ax4.scatter([0], [0], s=100, color='green', zorder=5)
    ax4.scatter([3.0], [2.0], s=140, color='red', marker='*', zorder=5)
    ax4.annotate('Home',   (0, 0),     xytext=(0.1, 0.1),  fontsize=8)
    ax4.annotate('Target', (3.0, 2.0), xytext=(3.05, 2.1), fontsize=8)
    jam_p  = mpatches.Patch(color=COLOR_JAM,  label='Jammed')
    norm_p = mpatches.Patch(color=COLOR_NORM, label='Normal')
    ax4.legend(handles=[norm_p, jam_p], fontsize=7, loc='upper left')
    ax4.set_title('2D Flight Trajectory'); ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
    ax4.grid(True, alpha=0.3); ax4.set_aspect('equal', 'box')

    # ── Row 2 ─────────────────────────────────────────────────────────

    # 2A  Position X/Y vs time
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.plot(t, df['pos_x'], label='X', color='royalblue')
    ax5.plot(t, df['pos_y'], label='Y', color='darkorange')
    _shade_jam(ax5, t, jam_mask)
    ax5.set_title('Position vs Time'); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('m')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # 2B  Altitude
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.plot(t, df['altitude'], color='purple')
    _shade_jam(ax6, t, jam_mask)
    ax6.set_title('Altitude'); ax6.set_xlabel('Time (s)'); ax6.set_ylabel('m')
    ax6.grid(True, alpha=0.3)

    # 2C  Battery
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.plot(t, df['battery'], color='green')
    ax7.axhline(20, color='red', linestyle='--', linewidth=1, label='Low (20%)')
    _shade_jam(ax7, t, jam_mask)
    ax7.set_title('Battery'); ax7.set_xlabel('Time (s)'); ax7.set_ylabel('%')
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    # 2D  RC Signal (RSSI)
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.plot(t, df['rc_rssi'], color='darkcyan')
    ax8.axhline(-100, color='red', linestyle='--', linewidth=1, label='Signal lost')
    _shade_jam(ax8, t, jam_mask)
    ax8.set_title('RC Signal Strength'); ax8.set_xlabel('Time (s)'); ax8.set_ylabel('RSSI (dBm)')
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3)

    # ── Row 3 ─────────────────────────────────────────────────────────

    # 3A  AI Confidence
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.scatter(t[jam_mask],  df['confidence'].values[jam_mask],
                s=5, c='red',  alpha=0.5, label='Jammed')
    ax9.scatter(t[~jam_mask], df['confidence'].values[~jam_mask],
                s=5, c='blue', alpha=0.2, label='Normal')
    ax9.axhline(0.8, color='green', linestyle='--', linewidth=1, label='80% threshold')
    ax9.set_ylim(0, 1.05)
    ax9.set_title('AI Confidence'); ax9.set_xlabel('Time (s)'); ax9.set_ylabel('Confidence')
    ax9.legend(fontsize=7); ax9.grid(True, alpha=0.3)

    # 3B  Stacked action probabilities (jammed window only)
    ax10 = fig.add_subplot(3, 4, 10)
    if jam_mask.any():
        jdf = df[jam_mask]
        jt  = jdf['t'].values
        ax10.stackplot(jt,
                       jdf['prob_continue'], jdf['prob_hover'],
                       jdf['prob_rth'],      jdf['prob_land'],
                       labels=['Continue', 'Hover', 'RTH', 'Land'],
                       alpha=0.85,
                       colors=['steelblue', 'orange', 'green', 'red'])
        ax10.set_title('Action Probs (Jammed Period)')
        ax10.set_xlabel('Time (s)'); ax10.set_ylabel('Probability')
        ax10.legend(fontsize=7, loc='upper right')
        ax10.grid(True, alpha=0.3)

    # 3C  IMU accelerometer
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.plot(t, df['accel_x'], label='Ax', alpha=0.7)
    ax11.plot(t, df['accel_y'], label='Ay', alpha=0.7)
    _shade_jam(ax11, t, jam_mask)
    ax11.set_title('Accelerometer'); ax11.set_xlabel('Time (s)'); ax11.set_ylabel('m/s²')
    ax11.legend(fontsize=8); ax11.grid(True, alpha=0.3)

    # 3D  Distance metrics
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.plot(t, df['dist_to_home'],   color='navy',   label='To Home')
    ax12.plot(t, df['dist_to_target'], color='crimson', label='To Target', linestyle='--')
    _shade_jam(ax12, t, jam_mask)
    ax12.set_title('Distance'); ax12.set_xlabel('Time (s)'); ax12.set_ylabel('m')
    ax12.legend(fontsize=8); ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    flight_path = out / 'flight_simulation.png'
    plt.savefig(str(flight_path), dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved: {flight_path}")

    # ── Training history (separate figure) ────────────────────────────
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Training History', fontweight='bold')
    a1.plot(history.history['loss'],     label='Train', color='steelblue')
    a1.plot(history.history['val_loss'], label='Val',   color='orange')
    a1.set_title('Loss'); a1.set_xlabel('Epoch')
    a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(history.history['accuracy'],     label='Train', color='steelblue')
    a2.plot(history.history['val_accuracy'], label='Val',   color='orange')
    a2.set_title('Accuracy'); a2.set_xlabel('Epoch')
    a2.set_ylim(0, 1.05); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout()
    history_path = out / 'training_history.png'
    plt.savefig(str(history_path), dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved: {history_path}")

    plt.close('all')
