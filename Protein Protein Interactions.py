#Analysing Protein-Protein Interaction Networks with GNNs (WORK IN PROGRESS)
#STILL NEED TO ADD DESCRIPTIONS OF ALL OF THE FUNCTIONS - FIX ERRORS
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

pyg_graph = from_networkx(G) #converts networkx KG to PyTorch geometric
pyg_graph.Brand #labels for node stored here

train_loader = NeighborLoader(data, input_nodes=pyg_graph.Brand,
                              shuffle=True, num_workers=os.cpu_count() - 2,
                              batch_size=1024, num_neighbors=[30] * 2)
total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                               batch_size=4096, shuffle=False,
                               num_workers=os.cpu_count() - 2)

#implement GNN model architecture (SAGE layers)
class SAGE(torch.nn.Module): #input dim = no of features in KG//out dim = no of predictions to make
    def __init__(self, input_dimension, hidden_dimension, output_dimension, n_layers = 2):
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()
    
        if n_layers == 1:
            self.layers.append(SAGEConv(input_dimension, output_dimension, normalize=False))
        elif n_layers == 2: 
            self.layers.append(SAGEConv(input_dimension, hidden_dimension, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_dimension))
            self.layers.append(SAGEConv(hidden_dimension, output_dimension, normalize=False))
        else:
           self.layers.append(SAGEConv(input_dimension, hidden_dimension, normalize=False))

        self.layers_bn.append(torch.nn.BatchNorm1d(hidden_dimension))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_dimension, hidden_dimension, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_dimension))
            self.layers.append(SAGEConv(hidden_dimension, output_dimension, normalize=False))

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index): #fn will
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=-1), torch.var(x)
    
    def inference(self, total_loader, device):
        xs = []
        var_ = []
        for batch in total_loader:
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())
        
        out_all = torch.cat(xs, dim=0)
        
        return out_all, var_


#parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAGE(data.x.shape[1], 256, data.num_classes, n_layers=2)#indim = 
model.to(device)
epochs = 1 #no of loops though dataset
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = ReduceLROnPlateau(optimizer, "max", patience=7)

#validates predictions
def test(model, device):
    evaluator = Evaluator(name=data)
    model.eval()
    out, var = model.inference(total_loader, device)
    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        "y_true": y_true[df_train],
        "y_pred": y_pred[df_train]
    })["acc"]
    val_acc = evaluator.eval({
        "y_true": y_true[df_validate],
        "y_pred": y_pred[df_validate]
    })["acc"]
    test_acc = evaluator.eval({
        "y_true": y_true[df_test],
        "y_pred": y_pred[df_test]
    })["acc"]
    return train_acc, val_acc, test_acc, torch.mean(torch.Tensor(var))

#train model
for epoch in range(1, epochs):
    model.train()
    pbar = tqdm(total=df_train.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = 0
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out, _ = model(batch.x.to(device), batch.edge_index.to(device))
        out = out[:batch_size]
        batch_y = batch.y[:batch_size].to(device)
        batch_y = torch.reshape(batch_y, (-1,))
        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        pbar.update(batch.batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    approx_acc = total_correct / df_train.size(0)
    train_acc, val_acc, test_acc, var = test(model, device)
    
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Var: {var:.4f}')


#compare results to existing knowledge





#Bibliography: 
    #Diego Lopez Yse: https://lopezyse.medium.com/make-interactive-knowledge-graphs-with-python-cfe520482197#:~:text=Create%20a%20knowledge%20graph&text=We%20have%20three%20lists%3A%20head,graph%20representation%20of%20the%20relationships.
    #Tiago Toledo Jr.: https://towardsdatascience.com/how-to-create-a-graph-neural-network-in-python-61fd9b83b54e
    #Stanford 2019: https://www.youtube.com/watch?v=-UjytpbqX4A
    #mdr223: https://gist.github.com/mdr223/fcd4063ffa7828eb5e83b584e3a23bb2

