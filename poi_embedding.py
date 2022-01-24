"""
The semantics preserved POI representation learning module
This python script could learn POI category representations (second-level categories) given POIs and a DT network
"""
from poi_network import *
from Variables import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import random
import math

'''get the poi DT network'''
data, le_poi_first_level, le_poi_second_level, poi_csv = create_poi_network()

'''start - annotated - get walks of 2nd category'''
walks = spatially_explicit_random_walk(data, length=length, n=n, p=p, q=q, weighted=True)
second_class_walks = []
for index, walk in enumerate(walks):
    if len(walk) > 1:
        second_class_walks.append([])
        for poi in walk:
            second_class = data.node_features()[poi][4]
            second_class_walks[-1].append(second_class)
'''end - annotated - get walks of 2nd category'''
second_class_number = len(le_poi_second_level.classes_)
vocab_size = second_class_number
print("vocab", vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""start - global negative sampling, i.e. the negative sample does not co-occur with target globally"""
global_second_class_walks = []
for i_temp in range(second_class_number):
    global_second_class_walks.append([])
for second_class_walk in second_class_walks:
    global_second_class_walks[int(second_class_walk[0])].extend(second_class_walk[1:])
for i_temp in range(second_class_number):
    global_second_class_walks[i_temp] = list(set(global_second_class_walks[i_temp]))
"""end - negative sampling"""


class POISet(tud.Dataset):
    def __init__(self, graph, second_class_number, second_class_walks):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(POISet, self).__init__()
        self.graph = graph
        self.second_class_number = second_class_number
        self.second_class_walks = second_class_walks

    def __len__(self):
        return len(self.second_class_walks)

    def __getitem__(self, index):

        center_place = self.second_class_walks[index][0]  # get the target

        positive_context = self.second_class_walks[index][1:]  #
        negative_places = list(set(range(self.second_class_number))-set(global_second_class_walks[int(center_place)]))
        if len(negative_places) < int(len(self.second_class_walks[index]) * k):
            negative_places_original_copy = negative_places.copy()
            for i_temp_1 in range(math.ceil(int(len(self.second_class_walks[index]) * k)/len(negative_places))-1):
                negative_places.extend(negative_places_original_copy)
        negative_context = random.sample(negative_places, int(len(self.second_class_walks[index]) * k))
        return torch.tensor(center_place).long(), \
               torch.LongTensor(positive_context), \
               torch.LongTensor(negative_context)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, second_class_hierarchy_pairs, le_lambda):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.le_lambda = torch.tensor(le_lambda).to(device)
        self.second_class_hierarchy_pairs = torch.tensor(second_class_hierarchy_pairs).to(device)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        """ input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, context size]
            neg_labels：negative words, [batch_size, context size * K)]

            return: loss, [batch_size]
        """
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss_graph = log_pos + log_neg
        loss_graph = (-loss_graph).mean()
        """start - categorical semantics"""
        l2_norm = torch.tensor(0, dtype=torch.float).to(device)
        for pair in self.second_class_hierarchy_pairs:
            embed_i = self.in_embed(pair[0])
            embed_j = self.in_embed(pair[1])
            l2_norm += torch.norm((embed_i-embed_j))
        loss_le = 0.5 * (l2_norm ** 2) * le_lambda
        """end - categorical semantics"""
        loss_combined = loss_graph + loss_le
        return loss_combined, loss_le

    def input_embedding(self):
        return self.in_embed.cpu().weight.detach().numpy()

    def clone_input_embedding(self):
        return self.in_embed.weight.clone().cpu()


dataset = POISet(data, second_class_number, second_class_walks)
print(dataset)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)
model = EmbeddingModel(vocab_size, embedding_size, second_class_hierarchy_pairs, le_lambda).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for e in range(100):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()
        loss, loss_le = model(input_labels, pos_labels, neg_labels)
        loss.backward()

        optimizer.step()
        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item(), 'loss_le', loss_le.item())

embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "poi_embedding.tensor")