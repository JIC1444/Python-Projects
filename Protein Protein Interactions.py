#Analysing Protein-Protein Interaction Networks with KGs and GNNs
import pandas as pd
import csv
import networkx as nx
import plotly.graph_objects as go

data = pd.read_csv("GNN Proj/Data/1849171.protein.links.v12.0.txt", 
                 sep=" ", names = ["protein1" , "protein2", "combined_score"])
#757118 rows, 3 cols

#visualise these interactions 
df = pd.DataFrame({"head": data["protein1"][0:200],
                    "relation": data["combined_score"][0:200], 
                    "tail": data["protein2"][0:200]})

G = nx.Graph()
for _, row in df.iterrows():
  G.add_edge(row["head"], row["tail"], label=row["relation"])

pos = nx.fruchterman_reingold_layout(G, k=0.5)

edge_traces = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode="lines",
        line=dict(width=0.5, color="gray"),
        hoverinfo="none"
     )
    edge_traces.append(edge_trace)

node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    mode="markers+text",
    marker=dict(size=10, color="lightblue"),
    text=[node for node in G.nodes()],
    textposition="top center",
    hoverinfo="text",
    textfont=dict(size=7)
)

edge_label_trace = go.Scatter(
    x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
    y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
    mode="text",
    text=[G[edge[0]][edge[1]]["label"] for edge in G.edges()],
    textposition="middle center",
    hoverinfo="none",
    textfont=dict(size=7)
)

layout = go.Layout(
    title="Knowledge Graph",
    titlefont_size=16,
    title_x=0.5,
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis_visible=False,
    yaxis_visible=False
)

fig = go.Figure(data=edge_traces + [node_trace, edge_label_trace], layout=layout)
fig.show()

#implement GNN model architecture using PyTorch - upload to GitHub 05/02/24


#train model


#compare results to existing knowledge

