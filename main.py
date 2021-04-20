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
        G.nodes[v[0]]['concern'] = 0
        # sick_neighbors_count = 0
        # for neighbor in v[1]:
        #     if neighbor in total_infected:
        #         sick_neighbors_count += 1
        # G.nodes[v[0]]['concern'] = sick_neighbors_count / graph.degree[v[0]]
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

    for v in G.adj.items():
        G.nodes[v[0]]['concern'] = 0
        if v[0] in total_infected:
            NI[0].append(v[0])
        else:
            S[0].append(v[0])
    # ----------------------------------------------------
    # Infection
    for i in range(1, iterations):
        died_now = set()
        for v in S[i-1]:
            for neighbor in G[v]:
                if neighbor in NI[i - 1]:
                    P = min(1, CONTAGION * G.get_edge_data(v, neighbor, default=0)['w'] * (
                            1 - G.nodes[v]['concern']))
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
            for neighbor in G[v]:
                if neighbor in total_infected:
                    sick_neighbors_count += 1
                elif neighbor in total_deceased:
                    dead_neighbors_count += 1
            G.nodes[v]['concern'] = (3 * dead_neighbors_count + sick_neighbors_count) / (graph.degree[v])
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


def LTM2(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    V, E, I = set(graph.nodes), {(e[0], e[1]): e[2]['w'] for e in graph.edges.data()}, [set(patients_0)]
    E.update({(e[1], e[0]): e[2]['w'] for e in graph.edges.data()})
    for t in range(1, iterations + 1):
        I_add, S = set(), V.difference(I[t - 1])
        for v in S:
            v_neighbors = set(graph.neighbors(v))
            v_infected_1, v_infected_2 = v_neighbors.intersection(I[t - 1]), v_neighbors.intersection(I[t - 2])
            concern = (len(v_infected_2) / len(v_neighbors)) if t != 1 else 0
            if (CONTAGION * sum({E[(v, u)] for u in v_infected_1})) >= (1 + concern):
                I_add.add(v)
        I.append(I[t - 1].union(I_add))

    print("the len is ", len(I[-1]))
    return I[-1]


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
    d1, d2 = compute_lethality_effect(G, T)
    plot_lethality_effect(d1, d2)
