"""
app.py — Streamlit browser dashboard for the Anti-Jamming Drone AI.

Usage:
    streamlit run app.py
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anti-Jamming Drone AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚁 Anti-Jamming Drone AI — Live Simulation")
st.markdown("Simulates a 30-second drone mission with GPS/RC jamming. The neural network AI takes autonomous decisions during the jamming window.")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mission Parameters")

    jam_start = st.slider("Jam Start (s)", 0.0, 25.0, 6.0, 0.5)
    jam_dur   = st.slider("Jam Duration (s)", 1.0, 20.0, 12.0, 0.5)
    duration  = st.slider("Mission Duration (s)", 20.0, 60.0, 30.0, 1.0)

    st.divider()
    st.header("Model")
    load_existing = st.checkbox("Use existing trained model", value=True)

    if not load_existing:
        samples = st.number_input("Training samples", 1000, 50000, 10000, 1000)
        epochs  = st.number_input("Epochs", 10, 200, 60, 10)
    else:
        samples, epochs = 10000, 60

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.caption("Models saved to `models/`  \nData cached to `data/`")

# ── Helper: load or train model ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_trained_model(load_existing: bool, samples: int, epochs: int):
    from src.model.drone_ai import DroneAI
    from src.model.data_generator import generate_training_data, load_cached

    if load_existing:
        try:
            ai = DroneAI.load(model_dir='models')
            return ai, None      # no history when loading
        except Exception:
            pass  # fall through to train

    cached = load_cached('data')
    if cached:
        X_seq, X_flat, y, feature_names = cached
    else:
        X_seq, X_flat, y, feature_names = generate_training_data(samples, cache_dir='data')

    ai = DroneAI(feature_names)
    history = ai.train(X_seq, y, epochs=epochs)
    ai.save(model_dir='models')
    ai.to_tflite(model_dir='models')
    return ai, history


def run_simulation(ai, jam_start, jam_dur, duration):
    from src.simulator.mission_simulator import MissionSimulator
    sim = MissionSimulator(ai)
    sim.run(duration=duration, dt=0.02, jam_start=jam_start, jam_dur=jam_dur)
    return sim


# ── Color helpers ─────────────────────────────────────────────────────────────
COLOR_NORM = '#4488ff'
COLOR_JAM  = '#ff4444'
COLOR_BG_JAM = 'rgba(255,68,68,0.10)'


def jam_shapes(df: pd.DataFrame, y0: float = 0, y1: float = 1):
    """Return Plotly shapes for jamming windows."""
    shapes = []
    jam = df['jammed'].values
    t   = df['t'].values
    in_j, t0 = False, 0.0
    for i, j in enumerate(jam):
        if j and not in_j:
            t0, in_j = t[i], True
        elif not j and in_j:
            shapes.append(dict(type='rect', x0=t0, x1=t[i], y0=y0, y1=y1,
                               fillcolor=COLOR_BG_JAM, line_width=0, layer='below'))
            in_j = False
    if in_j:
        shapes.append(dict(type='rect', x0=t0, x1=t[-1], y0=y0, y1=y1,
                           fillcolor=COLOR_BG_JAM, line_width=0, layer='below'))
    return shapes


# ── Main ──────────────────────────────────────────────────────────────────────
if run_btn:
    # 1. Load / train model
    with st.spinner("Loading / training model..."):
        try:
            ai, history = get_trained_model(load_existing, int(samples), int(epochs))
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    # 2. Run simulation
    with st.spinner("Running mission simulation..."):
        try:
            sim = run_simulation(ai, jam_start, jam_dur, duration)
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.stop()

    df = pd.DataFrame(sim.log)
    t  = df['t'].values
    jam_mask = df['jammed'].values.astype(bool)

    # ── KPI row ───────────────────────────────────────────────────────────────
    final = df.iloc[-1]
    bat_start = df.iloc[0]['battery']
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", "SUCCESS ✅" if sim.success else "INCOMPLETE ⚠️")
    c2.metric("Val Accuracy", f"{ai.history.history['val_accuracy'][-1]:.2%}" if ai.history else "N/A")
    c3.metric("Battery Used", f"{bat_start - final['battery']:.1f}%")
    c4.metric("Dist from Home", f"{final['dist_to_home']:.2f} m")
    c5.metric("AI Decisions (jammed)", str(jam_mask.sum()))

    st.divider()

    # ── Animated drone view ───────────────────────────────────────────────────
    st.subheader("Live Drone Map (Animated)")

    # Sample every 25 steps (~0.5 s per frame) to keep frame count manageable
    stride    = 25
    df_frames = df.iloc[::stride].reset_index(drop=True)

    WAYPOINTS = [[1.0, 0.3], [2.0, 1.5], [3.0, 2.0],   # outbound
                 [2.5, 0.8], [1.2, 0.2]]                 # return (excl. home)

    def _drone_color(row):
        if row['jammed']:
            action = row['action_name']
            if 'RTH'  in action: return '#ff8800'
            if 'LAND' in action: return '#ff0000'
            if 'HOVER' in action: return '#ffdd00'
            return '#ff4444'
        return '#00ccff'

    def _drone_label(row):
        name_map = {
            'CONTINUE_TO_TARGET': 'FLY',
            'HOVER_AT_TARGET':    'HOVER',
            'RETURN_HOME':        'RTH',
            'EMERGENCY_LAND':     'LAND',
        }
        action = name_map.get(row['action_name'], row['action_name'][:4])
        jam_tag = ' [JAM]' if row['jammed'] else ''
        return (f"t={row['t']:.1f}s  {action}{jam_tag}<br>"
                f"Bat:{row['battery']:.0f}%  Alt:{row['altitude']:.1f}m<br>"
                f"Conf:{row['confidence']:.0%}  Phase:{row['phase']}")

    # Build frames
    anim_frames = []
    for _, row in df_frames.iterrows():
        dc = _drone_color(row)
        drone_trace = go.Scatter(
            x=[row['pos_x']], y=[row['pos_y']],
            mode='markers+text',
            marker=dict(symbol='square', size=18, color=dc,
                        line=dict(color='white', width=2)),
            text=['🚁'], textposition='middle center',
            textfont=dict(size=10),
            name='Drone',
            hovertext=_drone_label(row),
            hoverinfo='text',
            showlegend=False,
        )
        # Jam radius circle (only during jamming)
        if row['jammed']:
            theta = np.linspace(0, 2*np.pi, 60)
            r = 1.2
            cx, cy = row['pos_x'], row['pos_y']
            jam_circle = go.Scatter(
                x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
                mode='lines', line=dict(color='rgba(255,50,50,0.4)', width=2, dash='dot'),
                fill='toself', fillcolor='rgba(255,50,50,0.07)',
                name='Jam zone', showlegend=False, hoverinfo='skip',
            )
            anim_frames.append(go.Frame(data=[drone_trace, jam_circle],
                                        name=str(row['t'])))
        else:
            anim_frames.append(go.Frame(data=[drone_trace],
                                        name=str(row['t'])))

    # Static base traces
    traj_x = df['pos_x'].values
    traj_y = df['pos_y'].values

    base_traces = []
    # Full trail (faint)
    base_traces.append(go.Scatter(
        x=traj_x, y=traj_y, mode='lines',
        line=dict(color='rgba(100,150,255,0.25)', width=1),
        name='Full path', showlegend=False, hoverinfo='skip',
    ))
    # Normal segments (blue)
    for i in range(1, len(df)):
        if not jam_mask[i]:
            base_traces.append(go.Scatter(
                x=traj_x[i-1:i+1], y=traj_y[i-1:i+1], mode='lines',
                line=dict(color=COLOR_NORM, width=2),
                showlegend=False, hoverinfo='skip',
            ))
    # Jammed segments (red)
    for i in range(1, len(df)):
        if jam_mask[i]:
            base_traces.append(go.Scatter(
                x=traj_x[i-1:i+1], y=traj_y[i-1:i+1], mode='lines',
                line=dict(color=COLOR_JAM, width=2),
                showlegend=False, hoverinfo='skip',
            ))
    # Waypoints
    wx, wy = zip(*WAYPOINTS)
    base_traces.append(go.Scatter(
        x=wx, y=wy, mode='markers',
        marker=dict(symbol='diamond', size=9, color='#aaaaaa',
                    line=dict(color='white', width=1)),
        name='Waypoints', hoverinfo='skip',
    ))
    # Home
    base_traces.append(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(symbol='square', size=16, color='#00cc44',
                    line=dict(color='white', width=2)),
        text=['H'], textposition='middle center',
        textfont=dict(size=9, color='white'),
        name='Home', hovertext='Home base', hoverinfo='text',
    ))
    # Target
    base_traces.append(go.Scatter(
        x=[3.0], y=[2.0], mode='markers+text',
        marker=dict(symbol='star', size=18, color='#ff4400',
                    line=dict(color='white', width=1)),
        text=['T'], textposition='middle center',
        textfont=dict(size=8, color='white'),
        name='Target', hovertext='Mission target', hoverinfo='text',
    ))
    # Initial drone position placeholder (overwritten by frames)
    first_row = df_frames.iloc[0]
    base_traces.append(go.Scatter(
        x=[first_row['pos_x']], y=[first_row['pos_y']],
        mode='markers+text',
        marker=dict(symbol='square', size=18, color=_drone_color(first_row),
                    line=dict(color='white', width=2)),
        text=['🚁'], textposition='middle center',
        textfont=dict(size=10),
        name='Drone', hoverinfo='skip', showlegend=False,
    ))

    slider_steps = [
        dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
             label=f"{float(f.name):.1f}s",
             method='animate')
        for f in anim_frames
    ]

    fig_anim = go.Figure(
        data=base_traces,
        frames=anim_frames,
        layout=go.Layout(
            height=480,
            margin=dict(l=10, r=10, t=10, b=60),
            xaxis=dict(title='X (m)', range=[-0.5, 4.0], showgrid=True,
                       gridcolor='rgba(200,200,200,0.3)'),
            yaxis=dict(title='Y (m)', range=[-0.5, 2.8], showgrid=True,
                       gridcolor='rgba(200,200,200,0.3)', scaleanchor='x'),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            font=dict(color='#cccccc'),
            legend=dict(orientation='h', y=-0.12, font=dict(size=10)),
            updatemenus=[dict(
                type='buttons', showactive=False,
                y=1.08, x=0.0, xanchor='left',
                buttons=[
                    dict(label='▶  Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=120, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸  Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')]),
                ],
            )],
            sliders=[dict(
                steps=slider_steps,
                currentvalue=dict(prefix='Time: ', suffix=' s',
                                  font=dict(size=12, color='#cccccc')),
                pad=dict(t=10), x=0.0, len=1.0,
                bgcolor='#1e2530',
                tickcolor='#555555',
                font=dict(color='#999999', size=9),
            )],
        ),
    )

    # Legend annotation box (top-right corner)
    fig_anim.add_annotation(
        x=3.9, y=2.75, xanchor='right', yanchor='top',
        text=(
            "<b>Legend</b><br>"
            "<span style='color:#00ccff'>■</span> Flying (normal)<br>"
            "<span style='color:#ff4444'>■</span> Flying (jammed)<br>"
            "<span style='color:#ffdd00'>■</span> Hovering<br>"
            "<span style='color:#ff8800'>■</span> RTH<br>"
            "<span style='color:#ff0000'>■</span> Emergency Land<br>"
            "<span style='color:#00cc44'>■</span> Home base<br>"
            "<span style='color:#ff4400'>★</span> Target<br>"
            "<span style='color:#aaaaaa'>◆</span> Waypoints"
        ),
        showarrow=False,
        font=dict(size=10, color='#cccccc'),
        align='left',
        bgcolor='rgba(20,25,35,0.85)',
        bordercolor='#444',
        borderwidth=1,
    )

    st.plotly_chart(fig_anim, use_container_width=True)
    st.caption("Press **▶ Play** to animate the mission, or drag the slider to scrub through any moment.")

    st.divider()

    # ── Row 1: Trajectory + AI decisions ─────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("2D Flight Trajectory")
        fig_traj = go.Figure()

        # Segments coloured by jammed state
        for i in range(1, len(df)):
            c = COLOR_JAM if jam_mask[i] else COLOR_NORM
            fig_traj.add_trace(go.Scatter(
                x=df['pos_x'].values[i-1:i+1],
                y=df['pos_y'].values[i-1:i+1],
                mode='lines', line=dict(color=c, width=2),
                showlegend=False, hoverinfo='skip',
            ))

        # Home + Target markers
        fig_traj.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text',
            marker=dict(color='green', size=12, symbol='circle'),
            text=['Home'], textposition='top right', name='Home'))
        fig_traj.add_trace(go.Scatter(x=[3.0], y=[2.0], mode='markers+text',
            marker=dict(color='red', size=14, symbol='star'),
            text=['Target'], textposition='top right', name='Target'))

        # Legend proxies
        fig_traj.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color=COLOR_NORM, width=3), name='Normal'))
        fig_traj.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color=COLOR_JAM, width=3), name='Jammed'))

        fig_traj.update_layout(
            xaxis_title='X (m)', yaxis_title='Y (m)',
            yaxis_scaleanchor='x', height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_b:
        st.subheader("AI Confidence During Jamming")
        if jam_mask.any():
            jdf = df[jam_mask]
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Scatter(
                x=jdf['t'], y=jdf['prob_continue'],
                stackgroup='one', name='Continue', fill='tonexty',
                line=dict(color='steelblue'),
            ))
            fig_conf.add_trace(go.Scatter(
                x=jdf['t'], y=jdf['prob_hover'],
                stackgroup='one', name='Hover', fill='tonexty',
                line=dict(color='orange'),
            ))
            fig_conf.add_trace(go.Scatter(
                x=jdf['t'], y=jdf['prob_rth'],
                stackgroup='one', name='RTH', fill='tonexty',
                line=dict(color='green'),
            ))
            fig_conf.add_trace(go.Scatter(
                x=jdf['t'], y=jdf['prob_land'],
                stackgroup='one', name='Land', fill='tonexty',
                line=dict(color='red'),
            ))
            fig_conf.update_layout(
                xaxis_title='Time (s)', yaxis_title='Probability',
                yaxis_range=[0, 1], height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation='h', y=-0.15),
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No jamming window in this simulation.")

    # ── Row 2: Telemetry ──────────────────────────────────────────────────────
    st.subheader("Telemetry")
    col1, col2, col3, col4 = st.columns(4)

    def time_fig(y, name, ylabel, color, df=df, extra_traces=None, yrange=None):
        shapes = jam_shapes(df, y0=float(df[y].min()), y1=float(df[y].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df[y], mode='lines',
                                  line=dict(color=color), name=name))
        if extra_traces:
            for tr in extra_traces:
                fig.add_trace(tr)
        fig.update_layout(
            xaxis_title='Time (s)', yaxis_title=ylabel,
            height=220, margin=dict(l=0, r=0, t=5, b=0),
            shapes=shapes,
            legend=dict(font_size=9, orientation='h', y=-0.35),
            yaxis=dict(range=yrange) if yrange else {},
        )
        return fig

    with col1:
        st.caption("Battery (%)")
        low_line = go.Scatter(x=df['t'], y=[20]*len(df), mode='lines',
                              line=dict(color='red', dash='dash', width=1), name='Low 20%')
        st.plotly_chart(time_fig('battery', 'Battery', '%', 'green',
                                 extra_traces=[low_line]), use_container_width=True)

    with col2:
        st.caption("RC Signal Strength (dBm)")
        lost_line = go.Scatter(x=df['t'], y=[-100]*len(df), mode='lines',
                               line=dict(color='red', dash='dash', width=1), name='Signal lost')
        st.plotly_chart(time_fig('rc_rssi', 'RSSI', 'dBm', 'darkcyan',
                                 extra_traces=[lost_line]), use_container_width=True)

    with col3:
        st.caption("Altitude (m)")
        st.plotly_chart(time_fig('altitude', 'Altitude', 'm', 'purple'),
                        use_container_width=True)

    with col4:
        st.caption("Distance (m)")
        shapes = jam_shapes(df)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(x=df['t'], y=df['dist_to_home'],
                                       mode='lines', line=dict(color='navy'), name='To Home'))
        fig_dist.add_trace(go.Scatter(x=df['t'], y=df['dist_to_target'],
                                       mode='lines', line=dict(color='crimson', dash='dash'),
                                       name='To Target'))
        fig_dist.update_layout(
            xaxis_title='Time (s)', yaxis_title='m',
            height=220, margin=dict(l=0, r=0, t=5, b=0),
            shapes=shapes,
            legend=dict(font_size=9, orientation='h', y=-0.35),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Row 3: Position X/Y + Accelerometer ──────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.caption("Position vs Time")
        shapes = jam_shapes(df)
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Scatter(x=df['t'], y=df['pos_x'],
                                      mode='lines', line=dict(color='royalblue'), name='X'))
        fig_pos.add_trace(go.Scatter(x=df['t'], y=df['pos_y'],
                                      mode='lines', line=dict(color='darkorange'), name='Y'))
        fig_pos.update_layout(
            xaxis_title='Time (s)', yaxis_title='m',
            height=220, margin=dict(l=0, r=0, t=5, b=0),
            shapes=shapes,
            legend=dict(font_size=9, orientation='h', y=-0.35),
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with col6:
        st.caption("Accelerometer")
        shapes = jam_shapes(df)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=df['t'], y=df['accel_x'],
                                      mode='lines', line=dict(color='teal'), name='Ax', opacity=0.8))
        fig_acc.add_trace(go.Scatter(x=df['t'], y=df['accel_y'],
                                      mode='lines', line=dict(color='coral'), name='Ay', opacity=0.8))
        fig_acc.update_layout(
            xaxis_title='Time (s)', yaxis_title='m/s²',
            height=220, margin=dict(l=0, r=0, t=5, b=0),
            shapes=shapes,
            legend=dict(font_size=9, orientation='h', y=-0.35),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── Row 4: Training history + Confusion matrix ────────────────────────────
    if ai.history:
        st.divider()
        st.subheader("Training History")
        col7, col8 = st.columns(2)

        with col7:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=ai.history.history['loss'],
                                           mode='lines', name='Train', line=dict(color='steelblue')))
            fig_loss.add_trace(go.Scatter(y=ai.history.history['val_loss'],
                                           mode='lines', name='Val', line=dict(color='orange')))
            fig_loss.update_layout(title='Loss', xaxis_title='Epoch',
                                    height=280, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_loss, use_container_width=True)

        with col8:
            fig_acc2 = go.Figure()
            fig_acc2.add_trace(go.Scatter(y=ai.history.history['accuracy'],
                                           mode='lines', name='Train', line=dict(color='steelblue')))
            fig_acc2.add_trace(go.Scatter(y=ai.history.history['val_accuracy'],
                                           mode='lines', name='Val', line=dict(color='orange')))
            fig_acc2.update_layout(title='Accuracy', xaxis_title='Epoch',
                                    yaxis_range=[0, 1.05], height=280,
                                    margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_acc2, use_container_width=True)

    if ai._cm is not None:
        st.subheader("Confusion Matrix")
        labels = ['CONTINUE', 'HOVER', 'RTH', 'LAND']
        fig_cm = px.imshow(ai._cm, text_auto=True, color_continuous_scale='Blues',
                            x=labels, y=labels, aspect='auto')
        fig_cm.update_layout(
            xaxis_title='Predicted', yaxis_title='Actual',
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Raw log table ─────────────────────────────────────────────────────────
    with st.expander("Raw simulation log"):
        st.dataframe(df.round(4), use_container_width=True, height=300)

else:
    st.info("Configure mission parameters in the sidebar and click **▶  Run Simulation**.")

    with st.expander("Project structure"):
        st.code("""
d:/DroneSimulation Project NIT/
├── app.py                          ← this file (Streamlit UI)
├── main.py                         ← CLI full pipeline
├── train.py                        ← CLI train only
├── simulate.py                     ← CLI simulate only
├── src/
│   ├── model/
│   │   ├── drone_ai.py             ← DroneAI class (Keras + TFLite)
│   │   └── data_generator.py       ← balanced 4-class training data
│   ├── simulator/
│   │   ├── mission_simulator.py    ← 30s waypoint mission
│   │   └── sensor_simulator.py     ← physics + 21-feature sensor model
│   └── visualization/
│       └── plotter.py              ← Matplotlib static PNG export
├── models/                         ← saved .keras / .tflite / scaler
├── data/                           ← cached X.npy, y.npy
└── outputs/                        ← static PNG outputs
        """, language='')
