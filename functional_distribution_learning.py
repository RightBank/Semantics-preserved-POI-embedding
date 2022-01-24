"""
The module for learning urban functional distributions, i.e. generating region embeddings from POI embeddings,
and mapping region embedding to functional distributions of each single region (the proportion of each function type
that a region bears)
This python script could evaluate the performance of our approach.
However, due to copyright reasons, we are unable to share ground truth data here, and instead we generated mock ground
truth data that is used in this script.
"""
import torch
from Variables import *
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import Set2Set
import pandas as pd
from poi_network import create_poi_network
from scipy.spatial.distance import canberra, chebyshev


class FunctionZoneDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(FunctionZoneDataset, self).__init__()
        self.data, self.slices = self.collate(data_list)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_file_name = "Emb/poi_category_embedding.tensor" # the learned embeddings should be fed here

emb = torch.load(emb_file_name, map_location=device)['in_embed.weight']
poi_csv = pd.read_csv("Data/POI_xiamen_island_2020_ZoneID.csv", encoding='utf-8')
network, le_poi_first_level, le_poi_second_level, poi_csv = create_poi_network()
zone_list = []
poi_zone_grouped = poi_csv.groupby(['ZoneID'])

ground_truth = torch.load("Data/mock_ground_truth.tensor")

for zone_id, pois in poi_zone_grouped:

    second_class_list = []
    second_class_emb_list = []
    for index, poi in pois.iterrows():
        second_class_list.append(poi['SecondLeve'])
        second_class_emb_list.append(list(emb[int(poi['SecondLeve'])]))
    zone = Data(x=torch.tensor(second_class_emb_list), y=ground_truth[zone_id], zone_id=zone_id)
    zone_list.append(zone)

function_zone_dataset = FunctionZoneDataset(zone_list)
print(function_zone_dataset)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.set2set = Set2Set(embedding_size, processing_steps=5)
        self.lin0 = torch.nn.Linear(embedding_size*2, embedding_size)
        self.lin1 = torch.nn.Linear(embedding_size, 10)

    def forward(self, data):

        out = self.set2set(data.x, data.batch).float()
        out = torch.tanh(self.lin0(out)).float()
        out = F.softmax(self.lin1(out), -1).float()
        return out.view(-1).float()


def train():
    model.train()
    loss_all = 0
    p_a = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_estimated = model(data)
        loss = F.kl_div(torch.log(y_estimated), data.y.float(), reduction='batchmean').float()
        p_a += (y_estimated - data.y).abs().sum()
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset), p_a / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0
    canberra_distance = 0
    chebyshev_distance = 0
    kl_dist = 0
    cos_dist = 0
    for data in loader:
        data = data.to(device)
        y_estimated = model(data)
        error += ((y_estimated - data.y).abs().sum())
        canberra_distance += canberra(y_estimated.cpu().detach().numpy(), data.y.cpu().detach().numpy())
        kl_dist += F.kl_div(torch.log(y_estimated), data.y.float(), reduction='batchmean').float()
        chebyshev_distance += chebyshev(y_estimated.cpu().detach().numpy(), data.y.cpu().detach().numpy())
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dist += cos(y_estimated, data.y)
    return error/len(loader.dataset), canberra_distance/len(loader.dataset), kl_dist/len(loader.dataset), \
           chebyshev_distance/len(loader.dataset), cos_dist/len(loader.dataset)


function_zone_dataset = function_zone_dataset.shuffle()
training_batch_size = 64
test_dataset = function_zone_dataset[:(len(function_zone_dataset) // 10 * 2)]

train_dataset = function_zone_dataset[(len(function_zone_dataset) // 10 * 2):]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

print(iter)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):

    loss, p_a = train()
    test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist = test(test_loader)

    if (epoch % 5 == 0) or (epoch == 1):
        print('Epoch: {:03d}, p_a: {:7f}, Loss: {:.7f}, '
              'Test MAE: {:.7f}, canberra_dist:{:.7f}, kl_dist:{:.7f}, chebyshev_distance:{:.7f}, cos_distance:{:.7f}'.
              format(epoch, p_a, loss, test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist))


