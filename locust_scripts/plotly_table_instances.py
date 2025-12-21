import plotly.graph_objects as go

# Organize data properly by columns
instances = [
    "ml.g4dn.xlarge",
    "ml.g5.xlarge",
    "ml.g6.xlarge",
    "ml.g4dn.xlarge",
    "ml.g5.xlarge",
    "ml.g6.xlarge",
    "ml.g4dn.xlarge",
    "ml.g5.xlarge",
    "ml.g6.xlarge",
]

users = [1, 1, 1, 10, 10, 10, 25, 25, 25]

latency = [170, 88, 87, 1200, 390, 380, 6000, 960, 940]

rps = [6.3, 12.1, 12.2, 8.3, 25.9, 26.1, 8.4, 26.1, 26.5]

gpu = ["79%", "28%", "37%", "91%", "75%", "81%", "95%", "73%", "82%"]

# Create table with properly separated columns
fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[180, 80, 120, 80, 90],
            header=dict(
                values=[
                    "<b>Instance</b>",
                    "<b>Users</b>",
                    "<b>Latency (ms)</b>",
                    "<b>RPS</b>",
                    "<b>GPU %</b>",
                ],
                fill_color="#34495E",
                align=["left", "center", "center", "center", "center"],
                font=dict(color="white", size=14, family="Arial, sans-serif"),
                height=45,
                line=dict(color="white", width=2),
            ),
            cells=dict(
                values=[
                    instances,
                    users,
                    latency,
                    rps,
                    gpu,
                ],  # Each column as separate list
                fill_color=[["#ECF0F1" if i % 2 == 0 else "white" for i in range(9)]],
                align=["left", "center", "center", "center", "center"],
                font=dict(color="#2C3E50", size=13, family="Arial, sans-serif"),
                height=38,
                line=dict(color="#BDC3C7", width=1),
            ),
        )
    ]
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>SageMaker Instance Performance Comparison</b><br><span style="font-size:13px;color:#7F8C8D">RF-DETR Model Load Testing Results</span>',
        x=0.5,
        xanchor="center",
        font=dict(size=18, color="#2C3E50", family="Arial, sans-serif"),
    ),
    width=750,
    height=550,
    margin=dict(l=30, r=30, t=100, b=30),
    paper_bgcolor="white",
)

# Show the table
fig.show()

# Save as PNG for Medium
fig.write_image("sagemaker_performance_table.png", width=750, height=550, scale=2)

print("✓ Table saved as 'sagemaker_performance_table.png'")
print("✓ Data is now properly separated into columns")
