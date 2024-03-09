import pandas as pd
import csv
import numpy as np
from IPython.display import display
import networkx as nx
import plotly.graph_objects as go

#want KG of form: head = bacteria1, relation = frequency , tail = location

phylum16S = pd.read_csv("Figures/phylum16S.txt", sep=" ") #importing data grouped by Daphne's fn
table = display(phylum16S) 
col_names = list(phylum16S.columns.values) #adds location as a header for the KG


phylum16S_head = []
phylum16S_relation = np.array([])
phylum16S_tail = []
locs = ["ALR","AHR","FAU","GUD","HOG","LAV","OVS","RAM","SKJ","ULV2","VES","VIC"] #VIC only has 7 repeats but rest of them have 8

#extracting data into suitable format df(3x(95x9)) = df(3x855)
for u in range(0,94): #appending bacteria phyla for each repeat
  for f in range(0,8):
    phylum16S_tail.append(locs[f])
  phylum16S_head = phylum16S_head + col_names
phylum16S_tail.pop()
p = []
for n in range(0,94):
  for g in col_names:
    roe = phylum16S[g][n]
    print(roe)
    p.append(roe)
print(p) #need to run this as sometimes it breaks??

print("head len",len(phylum16S_head))
print("p len", len(p))
print("tail len",len(phylum16S_tail))

n = len(phylum16S_head) - len(p)
for i in range(0,n):
  phylum16S_head.pop()
N = len(phylum16S_tail) - len(p)
for l in range(0,N):
  phylum16S_tail.pop()

phylum16S_graph = pd.DataFrame({"head":phylum16S_head, "relation":p, "tail":phylum16S_tail})
phylum16S_graph= phylum16S_graph[phylum16S_graph["relation"] != 0] #kills any rows with 0 as relation


#could do a graph of a graph? ie to include temp and precip.
G = nx.Graph() 
for _, row in phylum16S_graph.iterrows():
  G.add_edge(row["head"], row["tail"], label=row["relation"], weight = row["relation"])

pos = nx.fruchterman_reingold_layout(G, k=0.5)

edge_traces = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode="lines",
        line=dict(width=0.6, color="black"),
        hoverinfo="none"
     )
    edge_traces.append(edge_trace)

node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    mode="markers+text",
    marker=dict(size=14, color="lightgreen"),
    text=[node for node in G.nodes()],
    textposition="top center",
    hoverinfo="text",
    textfont=dict(size=14)
)

edge_label_trace = go.Scatter(
    x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
    y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
    mode="text",
    text=[G[edge[0]][edge[1]]["label"] for edge in G.edges()],
    textposition="middle center",
    hoverinfo="none",
    textfont=dict(size=14)
)

layout = go.Layout(
    title="Knowledge Graph of Norway Bacteria Data",
    titlefont_size=16,
    title_x=0.5,
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis_visible=False,
    yaxis_visible=False
)

fig = go.Figure(data=edge_traces + [node_trace, edge_label_trace], layout=layout)
fig.show() #will show KG in browser, KG is a "random" shape everytime

#GNN might not be the best way to go about analysing this graph, might be better to
#do more of a statistical approach to see which phyla correlate with others.
