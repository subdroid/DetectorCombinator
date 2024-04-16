import os
import numpy as np

from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math

import sys

class identify_neurons():
    def __init__(self):
        self.loc = os.path.join(os.getcwd(),"results_english")
        self.cluster_loc = os.path.join(os.getcwd(), f"cluster_lists_{str(sys.argv[1])}")
        # print(self.cluster_loc)
        self.n_clusters = int(sys.argv[1])

    def entropy(self,S):
        S = np.array(S)
        epsilon = 1e-10  # Small constant to avoid division by zero
        S = S + epsilon
        log_probs = np.where(S > 0, np.log2(S), 0)
        ent = -np.sum(S*log_probs)
        return ent
                        
    def calc_distance(self,vec1,vec2):
        C = []
        for c1 in range(len(vec1)):
            for c2 in range(len(vec1)):
                C1 = set(vec1[c1])
                C2 = set(vec2[c2])
                common_set = C1.intersection(C2)
                total_set  = C1.union(C2)
                dist = len(common_set) / len(total_set)
                C.append(dist)
        return self.entropy(C), np.std(C)

    def compare_save(self):
        lyr_dic = {'xglm-564M': 24,
                   'xglm-1.7B': 24,
                   'xglm-2.9B': 48,
                   'xglm-4.5B': 48,
                   'xglm-7.5B': 32
                   }
        overlap = {}
        stdmat = {}
        for model in os.listdir(self.cluster_loc):
            m_loc = os.path.join(self.cluster_loc,model)
            n_layers = lyr_dic[model]
            for cat in os.listdir(m_loc):
                if not cat in overlap.keys():
                    overlap[cat] = {}
                if not model in overlap[cat].keys():
                    overlap[cat][model] = {}
                if not cat in stdmat.keys():
                    stdmat[cat] = {}
                if not model in stdmat[cat].keys():
                    stdmat[cat][model] = {}
                c_loc = os.path.join(m_loc,cat)
                folders = os.listdir(c_loc)
                for i in tqdm(range(len(folders))):
                    for j in range(len(folders)):
                        if i!=j: # We do not want comparison between same test sets
                            l1 = folders[i]
                            l2 = folders[j]
                            item_name = f"{l1}-{l2}"
                            l1_loc = os.path.join(c_loc,l1)
                            l2_loc = os.path.join(c_loc,l2)
                            Dist = []
                            SD = []
                            for lid in range(n_layers):
                                L1 = os.path.join (l1_loc,f"{str(lid)}.json")
                                L2 = os.path.join (l2_loc,f"{str(lid)}.json")
                                try:
                                    with open(L1, 'r') as file:
                                        data1 = json.load(file)
                                    with open(L2, 'r') as file:
                                        data2 = json.load(file)
                                    dist,var = self.calc_distance(data1,data2)
                                    Dist.append(dist)       
                                    SD.append(var)
                                except FileNotFoundError:
                                    continue
                            overlap[cat][model][item_name] = Dist
                            stdmat[cat][model][item_name] = SD
        with open(f'overlap_clusters_{sys.argv[1]}.json', 'w') as fp:
            json.dump(overlap, fp)
        with open(f'std_clusters_{sys.argv[1]}.json', 'w') as fp:
            json.dump(stdmat, fp)


    def compare_plot(self):
        with open(f'overlap_clusters_{sys.argv[1]}.json', 'r') as fp:
            overlap = json.load(fp)
        plot_loc = os.path.join(os.getcwd(),f"cluster_overlap_plots_{sys.argv[1]}")
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)
        # Plot cluster similarity stats
        for cats in overlap.keys():
            category_data = overlap[cats]
            models = ['xglm-564M','xglm-1.7B','xglm-2.9B','xglm-7.5B','xglm-4.5B']
            
            save_loc1 = os.path.join(plot_loc,"multi_ling")
            if not os.path.exists(save_loc1):
                os.mkdir(save_loc1)
            save_loc2 = os.path.join(plot_loc,"En-X")
            if not os.path.exists(save_loc2):
                os.mkdir(save_loc2)
            save_loc3 = os.path.join(plot_loc,"En-En")
            if not os.path.exists(save_loc3):
                os.mkdir(save_loc3)
            
            for i, model in enumerate(models):
                model_name = model.replace(".","-")
                try:
                    data = category_data[model]
                    multiling = ["cs","de","fr","hi"]
                    colors = {"de":'indianred',"cs":'salmon',
                          "fr":'royalblue',"hi":'forestgreen',
                          "Ende":'orange'}
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-") 
                        rev_name = f"{lp_name[1]}-{lp_name[0]}"
                        if lp not in done and rev_name not in done:
                            if lp_name[0] in multiling and lp_name[1] in multiling:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp)
                            done.append(lp) 
                    plt.title(f"{cats}: {model}")
                    plt.legend(fontsize=8)
                    plt.xlabel('Layer')
                    plt.ylabel('Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc1,f"multiling_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()


                    multiling = ["Encs","Ende","Enfr","Enhi"]
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-")    
                        if lp_name[0] in multiling:
                            sl = lp_name[0][2:]
                            if lp_name[1]==sl:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp)
                            done.append(lp_name[0])
                    rest = []
                    for pairs in multiling:
                        if pairs not in done:
                            rest.append(pairs)
                    for lp in data.keys():
                        lp_name = lp.split("-")
                        if lp_name[1] in rest:
                            sl = lp_name[1][2:]
                            if lp_name[0]==sl:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp) 
                    plt.title(f"{cats}: {model}")
                    plt.legend()
                    plt.xlabel('Layer')
                    plt.ylabel('Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc2,f"En-X_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()
 
                    multiling = ["Encs","Ende","Enfr","Enhi"]
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-")
                        rev_name = f"{lp_name[1]}-{lp_name[0]}"
                        if lp_name[0] in multiling and lp_name[1] in multiling:
                            if lp_name[0]!=lp_name[1]:
                                if rev_name not in done:
                                    plot_data = data[lp]
                                    x = range(1, len(plot_data) + 1)
                                    plt.plot(x, plot_data, label=lp) 
                                    done.append(lp)
                    plt.title(f"{cats}: {model}")
                    plt.legend()
                    plt.xlabel('Layer')
                    plt.ylabel('Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc3,f"En-En_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()
                except KeyError:
                    continue        
    
    def compare_dev(self):
        with open(f'std_clusters_{sys.argv[1]}.json', 'r') as fp:
            overlap = json.load(fp)
        plot_loc = os.path.join(os.getcwd(),f"cluster_std_plots_{sys.argv[1]}")
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)
        # Plot cluster similarity stats
        for cats in overlap.keys():
            category_data = overlap[cats]
            models = ['xglm-564M','xglm-1.7B','xglm-2.9B','xglm-7.5B','xglm-4.5B']
            
            save_loc1 = os.path.join(plot_loc,"multi_ling")
            if not os.path.exists(save_loc1):
                os.mkdir(save_loc1)
            save_loc2 = os.path.join(plot_loc,"En-X")
            if not os.path.exists(save_loc2):
                os.mkdir(save_loc2)
            save_loc3 = os.path.join(plot_loc,"En-En")
            if not os.path.exists(save_loc3):
                os.mkdir(save_loc3)
            
            for i, model in enumerate(models):
                model_name = model.replace(".","-")
                try:
                    data = category_data[model]
                    multiling = ["cs","de","fr","hi"]
                    colors = {"de":'indianred',"cs":'salmon',
                          "fr":'royalblue',"hi":'forestgreen',
                          "Ende":'orange'}
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-") 
                        rev_name = f"{lp_name[1]}-{lp_name[0]}"
                        if lp not in done and rev_name not in done:
                            if lp_name[0] in multiling and lp_name[1] in multiling:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp)
                            done.append(lp) 
                    plt.title(f"{cats}: {model}")
                    plt.legend(fontsize=8)
                    plt.xlabel('Layer')
                    plt.ylabel('Std Dev of Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc1,f"std_multiling_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()


                    multiling = ["Encs","Ende","Enfr","Enhi"]
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-")    
                        if lp_name[0] in multiling:
                            sl = lp_name[0][2:]
                            if lp_name[1]==sl:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp)
                            done.append(lp_name[0])
                    rest = []
                    for pairs in multiling:
                        if pairs not in done:
                            rest.append(pairs)
                    for lp in data.keys():
                        lp_name = lp.split("-")
                        if lp_name[1] in rest:
                            sl = lp_name[1][2:]
                            if lp_name[0]==sl:
                                plot_data = data[lp]
                                x = range(1, len(plot_data) + 1)
                                plt.plot(x, plot_data, label=lp) 
                    plt.title(f"{cats}: {model}")
                    plt.legend()
                    plt.xlabel('Layer')
                    plt.ylabel('Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc2,f"std_En-X_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()
 
                    multiling = ["Encs","Ende","Enfr","Enhi"]
                    done = []
                    for lp in data.keys():
                        lp_name = lp.split("-")
                        rev_name = f"{lp_name[1]}-{lp_name[0]}"
                        if lp_name[0] in multiling and lp_name[1] in multiling:
                            if lp_name[0]!=lp_name[1]:
                                if rev_name not in done:
                                    plot_data = data[lp]
                                    x = range(1, len(plot_data) + 1)
                                    plt.plot(x, plot_data, label=lp) 
                                    done.append(lp)
                    plt.title(f"{cats}: {model}")
                    plt.legend()
                    plt.xlabel('Layer')
                    plt.ylabel('Inter-Cluster Similarity')
                    p_loc = os.path.join(save_loc3,f"std_En-En_{cats}_{model_name}.pdf")
                    plt.savefig(p_loc)
                    plt.close()
                    plt.clf()
                except KeyError:
                    continue        

    
    def make_dir(self,loc):
        if not os.path.exists(loc):
            os.mkdir(loc)

    def read_files(self):
        lyr_dic = {'xglm-564M': 24,
                   'xglm-1.7B': 24,
                   'xglm-2.9B': 48,
                   'xglm-4.5B': 48,
                   'xglm-7.5B': 32
                   }
        for model in os.listdir(self.loc):
            f_loc = os.path.join(self.loc,model)
            n_lyrs = lyr_dic[model]
            for cat in os.listdir(f_loc):
                cat_loc = os.path.join(f_loc,cat)
                # Browse through languages
                for lps in os.listdir(cat_loc):
                    act_dic = {}  
                    lp_loc = os.path.join(cat_loc,lps)
                    lp_name = lps
                    cluster_dic = {}
                    for lyr in tqdm(range(n_lyrs)):
                        L =[]
                        for sents in os.listdir(lp_loc):
                            sent_loc = os.path.join(lp_loc,sents)
                            fname = f"{lp_name}_lyr{lyr}.npy"
                            sl_loc = os.path.join(sent_loc,fname)
                            L.append(sl_loc)
                        arrays = []
                        for file_name in L:
                            array = np.load(file_name)
                            arrays.append(array)
                        combined_array = np.concatenate(arrays, axis=0)
                        # print(self.n_clusters)
                        # n_clusters = 16
                        # # n_clusters = 8
                        n_clusters = self.n_clusters
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        kmeans.fit(combined_array.T)
                        labels = kmeans.labels_
                        clusters = [[] for _ in range(n_clusters)]
                        X = np.arange(1, (len(labels)+1), 1, dtype=int)
                        for data_point, cluster_id in zip(X, labels):
                            clusters[cluster_id].append(str(data_point))
                        save_loc0 = os.path.join(os.getcwd(),self.cluster_loc)
                        self.make_dir(save_loc0)
                        save_loc1 = os.path.join(save_loc0,model)
                        self.make_dir(save_loc1)
                        save_loc2 = os.path.join(save_loc1,cat)
                        self.make_dir(save_loc2)
                        save_loc3 = os.path.join(save_loc2,lps)
                        self.make_dir(save_loc3)
                        file_name =f"{str(lyr)}.json"
                        save_loc = os.path.join(save_loc3,file_name)
                        # Write the list of lists to a JSON file
                        with open(save_loc, 'w') as file:
                                json.dump(clusters, file)
                                                            
idnfy = identify_neurons()
# idnfy.read_files()
# idnfy.compare_save()
idnfy.compare_plot() 
idnfy.compare_dev()
