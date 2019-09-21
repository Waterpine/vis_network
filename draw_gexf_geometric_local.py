import os
import os.path as osp
import math
import random
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# dataset PROTEINS MUTAG ENZYMES COLLAB NCI1 PTC_MR DD
# IMDB-MULTI IMDB-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K REDDIT-BINARY
# dataset_name = 'DD'
# layout circular random shell spring spectral
# layout_list = ['circular', 'random', 'shell', 'spring']
# layout_list = ['circular', 'random', 'shell', 'spring', 'spectral']
layout_list = ['spring']


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


def readnpy(dataset_name, net):
    a_true = []
    a_true_class = []
    a_false = []
    a_false_class = []
    if 'conv' in net:
        conv_flag = True
    else:
        conv_flag = False
    for filename in os.listdir("pic_num"):
        if conv_flag:
            if 'conv' not in filename:
                continue
        else:
            if 'conv' in filename:
                continue
        if filename.startswith(dataset_name) and net in filename and 'True.npy' in filename:
            temp = np.load(os.path.join("pic_num", filename))
            a_true.extend(temp)
        elif filename.startswith(dataset_name) and net in filename and 'True_class' in filename:
            temp = np.load(os.path.join("pic_num", filename))
            a_true_class.extend(temp)
        elif filename.startswith(dataset_name) and net in filename and 'False.npy' in filename:
            temp = np.load(os.path.join("pic_num", filename))
            a_false.extend(temp)
        elif filename.startswith(dataset_name) and net in filename and 'False_class' in filename:
            temp = np.load(os.path.join("pic_num", filename))
            a_false_class.extend(temp)
        else:
            pass
    a_true = np.array(a_true)
    a_true_class = np.array(a_true_class)
    a_false = np.array(a_false)
    a_false_class = np.array(a_false_class)
    return a_true, a_true_class, a_false, a_false_class


def main():
    # net ResGCN ResGFN ResGFN_conv0_fc2
    # datasets = ['PROTEINS', 'MUTAG', 'ENZYMES', 'COLLAB', 'NCI1', 'PTC_MR', 'DD',
    #             'IMDB-MULTI', 'IMDB-BINARY', 'REDDIT-MULTI-5K',
    #             'REDDIT-MULTI-12K', 'REDDIT-BINARY']
    datasets = ['PROTEINS', 'MUTAG', 'IMDB-MULTI', 'IMDB-BINARY']
    nets = ['ResGFN', 'ResGCN']
    with open('dict_graph_num.json') as load_f:
        dict_graph_num = json.load(load_f)
    for dataset_name in datasets:
        for net in nets:
            data_path = 'data_gexf/{}'.format(dataset_name)
            a_true, a_true_class, a_false, a_false_class = readnpy(dataset_name=dataset_name, net=net)
            class_num = dict_graph_num[dataset_name]
            with open(os.path.join(data_path, 'd.json')) as load_f:
                d = json.load(load_f)  # class to graph_id
            with open(os.path.join(data_path, 'd_reverse.json')) as load_f:
                d_reverse = json.load(load_f)  # graph_id to class
            with open(os.path.join(data_path, 'class_set.json')) as load_f:
                class_set = set(json.load(load_f))

            d_true = {}  # class to graph_id
            d_false = {}  # class to graph_id
            for i in range(len(a_true)):
                if d_reverse[str(a_true[i])] in d_true.keys():
                    assert d_reverse[str(a_true[i])] == a_true_class[i]
                    d_true[d_reverse[str(a_true[i])]].append(a_true[i])
                else:
                    assert d_reverse[str(a_true[i])] == a_true_class[i]
                    d_true[d_reverse[str(a_true[i])]] = []
                    d_true[d_reverse[str(a_true[i])]].append(a_true[i])

            for i in range(len(a_false_class)):
                if a_false_class[i] in d_false.keys():
                    assert d_reverse[str(a_false[i])] == a_false_class[i]
                    d_false[a_false_class[i]].append(a_false[i])
                else:
                    assert d_reverse[str(a_false[i])] == a_false_class[i]
                    d_false[a_false_class[i]] = []
                    d_false[a_false_class[i]].append(a_false[i])

            mkdir("picture")
            mkdir("picture/{}_pic".format(dataset_name))

            color_list = []
            if class_num != 0:
                colors = sns.color_palette("RdBu", class_num)
            else:
                colors = sns.color_palette("Blues", 1)
            for r, g, b in colors:
                r_str = hex(int(r * 255))[2:] if len(hex(int(r * 255))[2:]) == 2 else "0" + hex(int(r * 255))[2:]
                g_str = hex(int(g * 255))[2:] if len(hex(int(g * 255))[2:]) == 2 else "0" + hex(int(g * 255))[2:]
                b_str = hex(int(b * 255))[2:] if len(hex(int(b * 255))[2:]) == 2 else "0" + hex(int(b * 255))[2:]
                color_list.append("#" + r_str + g_str + b_str)
            print(color_list)

            graphs_ground_dict = {}
            graphs_true_dict = {}
            graphs_false_dict = {}
            for i in range(len(class_set)):
                select_ground_list = random.sample(d[str(i)], 5)
                select_true_list = []
                select_false_list = []
                true_list = d_true[i]
                false_list = d_false[i]
                result_true = Counter(true_list)
                result_false = Counter(false_list)
                true_list_sorted = sorted(result_true.items(), key=lambda x: x[1], reverse=True)
                false_list_sorted = sorted(result_false.items(), key=lambda x: x[1], reverse=True)
                for eval_num in range(5):
                    if eval_num >= len(true_list_sorted):
                        pass
                    else:
                        select_true_list.append(true_list_sorted[eval_num][0])
                    if eval_num >= len(false_list_sorted):
                        pass
                    else:
                        select_false_list.append(false_list_sorted[eval_num][0])

                graphs_ground_dict[i] = select_ground_list
                graphs_true_dict[i] = select_true_list
                graphs_false_dict[i] = select_false_list

            for layout in layout_list:
                plt.figure(figsize=(25, 5 * len(class_set)))
                plt.axis('off')
                for i in range(len(class_set)):
                    select_graphs = graphs_ground_dict[i]
                    for j in range(len(select_graphs)):
                        label_list = []
                        node_color_list = []
                        G = nx.read_gexf(os.path.join(data_path, '{}.gexf'.format(select_graphs[j])))
                        if layout == 'random':
                            pos = nx.random_layout(G)
                        elif layout == 'circular':
                            pos = nx.circular_layout(G)
                        elif layout == 'shell':
                            pos = nx.shell_layout(G)
                        elif layout == 'spring':
                            if dataset_name.startswith("IMDB"):
                                pos = nx.spring_layout(G, dim=2, iterations=200, scale=1.0)
                            else:
                                pos = nx.spring_layout(G, dim=2, k=0.5 / math.sqrt(len(G.nodes())),
                                                       iterations=200, scale=1.0)
                        else:
                            raise ValueError('layout error')
                        for node in G.nodes():
                            label_list.append(int(G.node[node]['value']))
                            node_color_list.append(colors[label_list[-1] - 1])
                        plt.subplot(len(class_set), 5, 5 * i + j + 1)
                        plt.axis('off')
                        nx.draw_networkx(G, pos=pos, labels=label_list, with_labels=False,
                                         node_color=node_color_list, node_size=120, linewidths=2.0)
                plt.savefig("picture/{}_pic/{}_{}_{}_Random.png".format(dataset_name, dataset_name, net, layout),
                            bbox_inches='tight')
                plt.savefig("picture/{}_pic/{}_{}_{}_Random.pdf".format(dataset_name, dataset_name, net, layout),
                            bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(25, 5 * len(class_set)))
                plt.axis('off')
                for i in range(len(class_set)):
                    select_graphs = graphs_false_dict[i]
                    for j in range(len(select_graphs)):
                        label_list = []
                        node_color_list = []
                        G = nx.read_gexf(os.path.join(data_path, '{}.gexf'.format(select_graphs[j])))
                        if layout == 'random':
                            pos = nx.random_layout(G)
                        elif layout == 'circular':
                            pos = nx.circular_layout(G)
                        elif layout == 'shell':
                            pos = nx.shell_layout(G)
                        elif layout == 'spring':
                            if dataset_name.startswith("IMDB"):
                                pos = nx.spring_layout(G, dim=2, iterations=200, scale=1.0)
                            else:
                                pos = nx.spring_layout(G, dim=2, k=0.5 / math.sqrt(len(G.nodes())),
                                                       iterations=200, scale=1.0)
                        else:
                            raise ValueError('layout error')
                        for node in G.nodes():
                            label_list.append(int(G.node[node]['value']))
                            node_color_list.append(colors[label_list[-1] - 1])
                        plt.subplot(len(class_set), 5, 5 * i + j + 1)
                        plt.axis('off')
                        nx.draw_networkx(G, pos=pos, labels=label_list, with_labels=False,
                                         node_color=node_color_list, node_size=120, linewidths=2.0)
                plt.savefig("picture/{}_pic/{}_{}_{}_mis.png".format(dataset_name, dataset_name, net, layout),
                            bbox_inches='tight')
                plt.savefig("picture/{}_pic/{}_{}_{}_mis.pdf".format(dataset_name, dataset_name, net, layout),
                            bbox_inches='tight')
                plt.close()


if __name__ == '__main__':
    main()
