from hydrobr.get_data import ANA
from hydrobr.graphics import Plot

# 1) Find precipitation gauges (change state/city as you like)
stations = ANA.list_prec(state="Minas Gerais", city="Belo Horizonte", source="ANA")
print(f"Found {len(stations)} precipitation stations")
print(stations.head())


# 2) Plot interactive station map (requires a Mapbox token)
MAPBOX_TOKEN = "pk.eyJ1IjoibWFyY3Vzbm9icmVnYSIsImEiOiJjbWU5OTF0NnMwb3dqMmtvaW9reWRkOXM2In0.gPJCXTxoKtlx5TOn7cXQzQ"  # get one at https://account.mapbox.com/access-tokens/
fig = Plot.spatial_stations(stations, mapbox_access_token=MAPBOX_TOKEN)
fig.show()

