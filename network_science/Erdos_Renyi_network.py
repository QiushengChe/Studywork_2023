import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_matrix(N, p):
    adjacency_matrix = np.zeros((N, N))
    isolate = np.zeros(N)  # the point is isolated or not, 0 for isolate

    for i in range(N):
        for j in range(i, N):
            if p >= np.random.rand():
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                isolate[i] = 1
                isolate[j] = 1

    return adjacency_matrix


def connect_judge(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    S_connect = 0
    Q_dots = np.zeros(n)
    edge_cnt = 1

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == 1:
                if Q_dots[i] == Q_dots[j]:
                    if Q_dots[i] == 0:
                        Q_dots[i] = edge_cnt
                        Q_dots[j] = edge_cnt
                        edge_cnt += 1
                        S_connect += 1
                else:
                    if Q_dots[i] == 0:
                        Q_dots[i] = Q_dots[j]
                    elif Q_dots[j] == 0:
                        Q_dots[j] = Q_dots[i]
                    else:
                        temp = Q_dots[i]
                        Q_dots[Q_dots == temp] = Q_dots[j]
                        S_connect -= 1
    return S_connect, Q_dots


if __name__ == '__main__':
    # initial
    combi_cnt = 0
    for N in [1e2, 1e3, 2e3]:  # number of nodes, 1e2 / 1e3 / 2e3
        for p in [1e-5, 1e-4, 1e-3, 1e-2]:
            # possibility of connect between two nodes
            combi_cnt += 1
            # generate nodes and edge
            adjacency_matrix = generate_matrix(int(N), p)
            (S_connect, Q_dots) = connect_judge(adjacency_matrix)
            connect_list = np.unique(Q_dots)
            if connect_list[0] == 0:
                np.delete(connect_list, 0)

            max_connect_num = np.zeros(connect_list.shape[0])
            for i in range(connect_list.shape[0]):
                max_connect_num[i] = len(np.nonzero(Q_dots == connect_list[i]))

            isolation = len(np.nonzero(Q_dots == 0))
            # plot figure and save data
            adjacency_graph = nx.Graph(adjacency_matrix)
            plt.figure(combi_cnt)
            if S_connect < 1e2:
                width_line = 2
            elif S_connect < 1e3:
                width_line = 1e-4
            else:
                width_line = 1e-6
            nx.draw(adjacency_graph, pos=nx.spring_layout(adjacency_graph),
                    node_color='black',
                    node_size=10,
                    width=width_line,
                    edge_color='lightcyan')
            plt.savefig("./figure/" + '%d' % combi_cnt + ".png")
            print("Combination:", combi_cnt,
                  "N:", N, "p:", p,
                  "max_connect:", max(max_connect_num),
                  "isolate:", isolation)
            # plt.show()
            debug_point = 1
