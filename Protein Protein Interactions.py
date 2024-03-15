#Analysing Protein-Protein Interaction Networks with GNNs (ALMOST FINISHED)
#ERROR WITH DATASET TYPES LINE 154
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import networkx as nx
import plotly.graph_objects as go
import torch
from torch_geometric.utils.convert import to_networkx, from_networkx
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import os.path as osp
import time
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import f1_score



data = pd.read_csv("AstraZeneca GNN Proj/Data/1849171.protein.links.v12.0.txt", 
                 sep=" ", names = ["protein1" , "protein2", "combined_score"])
#757118 rows, 3 cols

#train validate test split data
df_train_t, df_test_temp = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
df_validate_t, df_test_t = train_test_split(df_test_temp, test_size=0.66) #splits test 30% to val 10% and train to 20%


df_train = pd.DataFrame({"head": df_train_t["protein1"],
                    "relation": df_train_t["combined_score"], 
                    "tail": df_train_t["protein2"]})

df_test = pd.DataFrame({"head": df_test_t["protein1"],
                    "relation": df_test_t["combined_score"], 
                    "tail": df_test_t["protein2"]})

df_validate = pd.DataFrame({"head": df_validate_t["protein1"],
                    "relation": df_validate_t["combined_score"], 
                    "tail": df_validate_t["protein2"]})




#visualise these interactions 
G = nx.Graph()
for _, row in df_train.iterrows():
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
#fig.show() #shows KG in webpage


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = train_dataset.to(device, 'edge_index')

model = GraphSAGE(
    data.num_node_features,
    hidden_channels=64,
    num_layers=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc


times = []
for epoch in range(1, 51):
    start = time.time()
    loss = train()
    val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

