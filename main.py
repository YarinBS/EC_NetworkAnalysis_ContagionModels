from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    total_infected = set(patients_0)
    not_infected = set(graph.nodes).difference(total_infected)
    #  STEP of Concern Update
    for v in graph.nodes:
        graph.nodes[v]['concern'] = 0
    # --------------------------------------------------------------------------
    for i in range(iterations):
        # Step of New Infected
        for v in not_infected:
            edges_w_sum = 0
            for neighbor in graph[v]:
                if neighbor in total_infected:
                    edges_w_sum += graph.get_edge_data(v, neighbor, default=0)['w']
            if CONTAGION * edges_w_sum >= 1 + graph.nodes[v]['concern']:
                total_infected.add(v)
                not_infected.remove(v)
        # --------------------------------------------
        #         Update S
        # not_infected = set(graph.nodes).difference(total_infected)
        # --------------------------------------------
        # UPDATE CONCERN******************************
        for v in not_infected:
            sick_neighbors_count = 0
            for neighbor in graph[v]:
                if neighbor in total_infected:
                    sick_neighbors_count += 1
            graph.nodes[v]['concern'] = sick_neighbors_count / graph.degree[v]
    # ------------------------------------------------------------------
    # print(len(total_infected))
    return total_infected


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    # initiation
    total_deceased = set()
    for pat in patients_0:
        rand = np.random.random()
        if rand < LETHALITY:
            total_deceased.add(pat)
            patients_0.remove(pat)
    total_infected = set(patients_0)

    S = [[] for i in range(iterations)]
    NI = [[] for i in range(iterations)]

    for v in graph.adj.items():
        graph.nodes[v[0]]['concern'] = 0
        if v[0] in total_infected:
            NI[0].append(v[0])
        else:
            S[0].append(v[0])
    # ----------------------------------------------------
    # Infection
    for i in range(1, iterations):
        died_now = set()
        for v in S[i - 1]:
            for neighbor in graph[v]:
                if neighbor in NI[i - 1]:
                    P = min(1, CONTAGION * graph.get_edge_data(v, neighbor, default=0)['w'] * (
                            1 - graph.nodes[v]['concern']))
                    rand = np.random.random()
                    if rand < P:
                        rand = np.random.random()
                        if rand < LETHALITY:
                            total_deceased.add(v)
                            died_now.add(v)
                        else:
                            NI[i].append(v)
                        continue
        total_infected.update(NI[i])
        S[i] = list(set(S[i - 1]).difference(total_infected.union(died_now)))
        # UPDATES CONCERN
        for v in S[i]:
            sick_neighbors_count = 0
            dead_neighbors_count = 0
            for neighbor in graph[v]:
                if neighbor in total_infected:
                    sick_neighbors_count += 1
                elif neighbor in total_deceased:
                    dead_neighbors_count += 1
            if graph.degree[v] == 0:
                graph.nodes[v]['concern'] = 0
            else:
                graph.nodes[v]['concern'] = (3 * dead_neighbors_count + sick_neighbors_count) / (graph.degree[v])
        # --------------------
    print("inectedAndAlive ", len(total_infected))
    print("died: ", len(total_deceased))
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
        counter_alive = 0
        counter_dead = 0
        LETHALITY = l
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            s_alive, s_dead = ICM(graph, list(patients_0), t)
            counter_alive += len(s_alive)
            counter_dead += len(s_dead)
        mean_deaths[l] = counter_dead / 30
        mean_infected[l] = counter_alive / 30

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    plt.plot(list(mean_deaths.keys()), list(mean_deaths.values()))
    plt.plot(list(mean_infected.keys()), list(mean_infected.values()))
    plt.show()


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    people_to_vaccinate = []
    # T = networkx.algorithms.maximum_spanning_tree(graph)
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:100]
    l1 = [node[0] for node in sorted_nodes]
    g2 = graph.subgraph(l1)
    l2= networkx.closeness_centrality(g2)
    sorted_by_closness = sorted(l2.items(), key=lambda item: item[1], reverse=True)[:70]
    l3 = [i[0] for i in sorted_by_closness]
    g3 = graph.subgraph(l3)
    contenders = networkx.betweenness_centrality_subset(g3, l3, l3, normalized=True)
    sorted_nodes = sorted(contenders.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [i[0] for i in sorted_nodes]
    patientsRand = []
    graph.remove_nodes_from([n for n in graph if n in set(people_to_vaccinate)])
    for i in range (50):
        patientsRand.append(np.random.choice(graph.nodes()))
    # ICM(graph, patientsRand, 6)
    return people_to_vaccinate


def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    """
    The following heuristic for Part C is simply taking the top 50 friendly people;
     that is, it returns the top 50 nodes in the graph with the highest degree.
    """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:100]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = 0.8
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
    patients_0 = [x[0] for x in pd.read_csv('patients0.csv', header=None).values]
    T = 6
    # LTM(G, patients_0, T)
    # LTM2(G, patients_0, T)
    # d1, d2 = compute_lethality_effect(G, T)
    # print(d1)
    # print(d2)
    # plot_lethality_effect(d1, d2)
    choose_who_to_vaccinate(G)
