from Variables import *
import pandas as pd
from sklearn import preprocessing

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk


def create_poi_network(poi_csv_path='Data/POI_xiamen_island_2020_ZoneID.csv',
                       edge_csv_path='Data/xiamen_edge_index_weight.csv'):
    r"""The function that constructs a graph using stellargraph.
        Args:
            poi_csv_path (str): the path of the csv file of poi data.
            edge_csv_path (str): the path of the csv file of edge index and pre-assigned two transition biases.
        """
    poi_csv = pd.read_csv(poi_csv_path, encoding='utf-8')
    edge_csv = pd.read_csv(edge_csv_path, encoding='utf-8')

    le_poi_first_level = preprocessing.LabelEncoder()
    le_poi_second_level = preprocessing.LabelEncoder()

    poi_csv['FirstLevel'] = le_poi_first_level.fit_transform(poi_csv['FirstLevel'].values)
    poi_csv['SecondLeve'] = le_poi_second_level.fit_transform(poi_csv['SecondLeve'].values)

    graph = StellarGraph(nodes=poi_csv, edges=edge_csv)
    print(graph.info())
    return graph, le_poi_first_level, le_poi_second_level, poi_csv


def spatially_explicit_random_walk(graph, length, n, p, q, weighted, seed=None):
    print("walks started")
    rw = BiasedRandomWalk(graph)
    walks = rw.run(
        nodes=graph.nodes(),  # root nodes
        length=length,  # maximum length of a random walk
        n=n,  # number of random walks per root node
        p=p,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=q,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=weighted,  # for weighted random walks
        seed=seed,  # random seed fixed for reproducibility
    )
    print("walks completed")
    return walks


