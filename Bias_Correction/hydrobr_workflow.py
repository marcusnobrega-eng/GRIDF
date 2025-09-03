from hydrobr.get_data import ANA
from hydrobr.graphics import Plot
import pandas as pd

# 1) Search for ANA flow stations in Minas Gerais (change state if you want)
stations = ANA.list_flow(state="Minas Gerais", source="ANA")
print("Found stations:", len(stations))
print(stations.head())

# 2) Take first 3 station codes from the search
codes = stations['Code'].astype(str).head(3).tolist()
print("Using station codes:", codes)

# 3) Download daily flow data for these stations
flows = ANA.flow(codes, only_consisted=False)
print("Flow data shape:", flows.shape)
print(flows.head())

# 4) Make a Flow Duration Curve (interactive Plotly figure)
fig_fdc = Plot.fdc(flows, y_log_scale=True)
fig_fdc.show()  # This opens the plot in your browser

# Optional: Save the data to CSV
flows.to_csv("flows_minimal_example.csv", index=True)
print("Saved daily flow data to flows_minimal_example.csv")
