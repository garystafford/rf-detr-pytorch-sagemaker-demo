import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data
instances = ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.g6.xlarge"]
users = [1, 10, 25]

# Latency data (ms)
latency_data = {
    "ml.g4dn.xlarge": [170, 1200, 6000],
    "ml.g5.xlarge": [88, 390, 960],
    "ml.g6.xlarge": [87, 380, 940],
}

# RPS data
rps_data = {
    "ml.g4dn.xlarge": [6.3, 8.3, 8.4],
    "ml.g5.xlarge": [12.1, 25.9, 26.1],
    "ml.g6.xlarge": [12.2, 26.1, 26.5],
}

# GPU utilization data (%)
gpu_data = {
    "ml.g4dn.xlarge": [79, 91, 95],
    "ml.g5.xlarge": [28, 75, 73],
    "ml.g6.xlarge": [37, 81, 82],
}

# Create subplots
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        "<b>Latency vs Users (Less is better)</b>",
        "<b>RPS vs Users (More is better)</b>",
        "<b>GPU Utilization vs Users (Less is better)</b>",
    ),
    vertical_spacing=0.12,
)

# Define colors with RGBA for transparency
colors = {
    "ml.g4dn.xlarge": "rgba(31, 119, 180, 0.7)",  # blue with 70% opacity
    "ml.g5.xlarge": "rgba(255, 127, 14, 0.7)",  # orange with 70% opacity
    "ml.g6.xlarge": "rgba(44, 160, 44, 0.7)",  # green with 70% opacity
}

# Add latency traces
for instance in instances:
    fig.add_trace(
        go.Scatter(
            x=users,
            y=latency_data[instance],
            name=instance,
            mode="lines+markers",
            line=dict(color=colors[instance], width=2.5),
            marker=dict(
                size=9, symbol="circle", color=colors[instance], line=dict(width=0)
            ),
            legendgroup=instance,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

# Add RPS traces
for instance in instances:
    fig.add_trace(
        go.Scatter(
            x=users,
            y=rps_data[instance],
            name=instance,
            mode="lines+markers",
            line=dict(color=colors[instance], width=2.5),
            marker=dict(
                size=9, symbol="circle", color=colors[instance], line=dict(width=0)
            ),
            legendgroup=instance,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

# Add GPU traces
for instance in instances:
    fig.add_trace(
        go.Scatter(
            x=users,
            y=gpu_data[instance],
            name=instance,
            mode="lines+markers",
            line=dict(color=colors[instance], width=2.5),
            marker=dict(
                size=9, symbol="circle", color=colors[instance], line=dict(width=0)
            ),
            legendgroup=instance,
            showlegend=False,
        ),
        row=3,
        col=1,
    )

# Update layout
fig.update_layout(
    height=1450,
    width=900,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.06,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0)",
        font=dict(size=12),
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=13, color="#555"),
    margin=dict(t=160, b=70, l=80, r=50),
)

# Update x-axes
fig.update_xaxes(
    title_text="<b>Users</b>",
    title_font=dict(size=14),
    showgrid=True,
    gridwidth=0.5,
    gridcolor="#e0e0e0",
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="#333",
    range=[-1, 27],
    dtick=5,
)

# Update y-axes
fig.update_yaxes(
    title_text="<b>Latency (ms)</b>",
    title_font=dict(size=14),
    showgrid=True,
    gridwidth=0.5,
    gridcolor="#e0e0e0",
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="#333",
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="<b>RPS</b>",
    title_font=dict(size=14),
    showgrid=True,
    gridwidth=0.5,
    gridcolor="#e0e0e0",
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="#333",
    row=2,
    col=1,
)

fig.update_yaxes(
    title_text="<b>GPU Utilization (%)</b>",
    title_font=dict(size=14),
    showgrid=True,
    gridwidth=0.5,
    gridcolor="#e0e0e0",
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="#333",
    row=3,
    col=1,
)

# Add title annotation
fig.add_annotation(
    text="<b>Latency, RPS, and GPU Utilization with Increased User Load Across ML Instances</b>",
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.09,
    showarrow=False,
    font=dict(size=17, color="#333"),
    xanchor="center",
    yanchor="top",
)

# Show the plot
fig.show()

# ========================================
# SAVE AS IMAGE FOR MEDIUM POST
# ========================================

# Install kaleido if you haven't already:
# pip install kaleido

# Save as PNG (best for Medium - high quality, widely supported)
fig.write_image(
    "sagemaker_performance_comparison.png",
    width=900,
    height=1450,
    scale=2,  # 2x resolution for retina displays
)

# Alternative: Save as SVG (vector format - scales perfectly but larger file)
# fig.write_image("sagemaker_performance_comparison.svg")

# Alternative: Save as HTML (interactive but may not embed in Medium)
# fig.write_html("sagemaker_performance_comparison.html")

print("✓ Chart saved as 'sagemaker_performance_comparison.png'")
print("✓ Upload this image to your Medium post")
