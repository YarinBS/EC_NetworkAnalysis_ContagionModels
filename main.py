from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    total_infected = set(patients_0)
    not_infected = set()
    #  STEP of Concern Update
    for v in G.adj.items():
        sick_neighbors_count = 0
        for neighbor in v[1]:
            if neighbor in total_infected:
                sick_neighbors_count += 1
        G.nodes[v[0]]['concern'] = sick_neighbors_count / graph.degree[v[0]]
    # --------------------------------------------------------------------------
    for i in range(iterations):
        # Step of New Infected
        for v in G.adj.items():
            edges_w_sum = 0
            for neighbor in v[1]:
                if neighbor in total_infected:
                    edges_w_sum += G.get_edge_data(v[0], neighbor, default=0)['w']
            if CONTAGION * edges_w_sum >= 1 + G.nodes[v[0]]['concern']:
                total_infected.add(v[0])
        # --------------------------------------------
        #         Update S
        not_infected = set(G.nodes).difference(total_infected)
        # --------------------------------------------
        # UPDATE CONCERN******************************
        for v in G.adj.items():
            if v[0] in not_infected:
                sick_neighbors_count = 0
                for neighbor in v[1]:
                    if neighbor in total_infected:
                        sick_neighbors_count += 1
                G.nodes[v[0]]['concern'] = sick_neighbors_count / graph.degree[v[0]]
    # ------------------------------------------------------------------
    print(total_infected)
    return total_infected


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_infected = set(patients_0)
    total_deceased = set()
    # TODO implement your code here
    return total_infected, total_deceased


def plot_degree_histogram(histogram: Dict):
    histogram = dict(sorted(histogram.items()))
    # plt.figure(figsize=(20, 20))
    plt.bar(range(len(histogram)), list(histogram.values()), align='center')
    # plt.xticks(range(len(histogram)), list(histogram.keys()))
    plt.show()


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    histogram = {}

    for node in graph.degree:
        if node[1] in histogram:
            histogram[node[1]] += 1
        else:
            histogram[node[1]] = 1
    # TODO implement your code here
    return histogram


def build_graph(filename: str) -> networkx.Graph:
    Data = open(filename, "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = networkx.Graph()
    G = networkx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                nodetype=int, data=(('w', float),))
    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    denominator = 0
    for value in graph.adj.values():
        denominator += comb(len(value), 2)
    tri_dict = networkx.triangles(graph)
    numerator = 3 * sum(tri_dict.values()) / 3
    cc = numerator / denominator
    print(cc)
    return cc


def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            # TODO implement your code here

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    # TODO implement your code here
    ...


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    people_to_vaccinate = []
    # TODO implement your code here
    return people_to_vaccinate


def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    """
    The following heuristic for Part C is simply taking the top 50 friendly people;
     that is, it returns the top 50 nodes in the graph with the highest degree.
    """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = 1
LETHALITY = .15


def comb(n, k):
    if n < k:
        return 0
    return int((np.lib.math.factorial(n) / (np.lib.math.factorial(k) * np.lib.math.factorial(n - k))))


def show_data(filename: str) -> networkx.Graph:
    G = build_graph(filename=filename)
    hist = calc_degree_histogram(G)
    plot_degree_histogram(hist)
    clustering_coefficient(G)


if __name__ == "__main__":
    # print("A1:")
    # show_data("PartA1.csv")
    # print("A2:")
    # show_data("PartA2.csv")
    G = build_graph("PartB-C.csv")
    l1 = [14270, 12029, 1, 25564]
    T = 2
    LTM(G, l1, T)
