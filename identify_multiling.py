import os
import numpy as np

from sklearn.cluster import KMeans
import json

class identify_neurons():
    def __init__(self):
        self.loc = os.path.join(os.getcwd(),"results_english")
        self.cluster_loc = os.path.join(os.getcwd(), "cluster_lists")
    def compare(self):
        lyr_dic = {'xglm-564M': 24,
                   'xglm-1.7B': 24,
                   'xglm-2.9B': 48,
                   'xglm-4.5B': 48,
                   'xglm-7.5B': 32
                   }
        for model in os.listdir(self.cluster_loc):
            m_loc = os.path.join(self.cluster_loc,model)
            n_layers = lyr_dic[model]
            for cat in os.listdir(m_loc):
                c_loc = os.path.join(m_loc,cat)
                for lid in range(n_layers):
                    C = []
                    for lp in os.listdir(c_loc):
                        l_loc  = os.path.join(c_loc,lp)
                        try:
                            l_file = os.path.join (l_loc,f"{str(lid)}.json")
                            with open(l_file, 'r') as file:
                                data = json.load(file)
                                C.append(data)
                        except:
                            continue
                    for lp in C:
                        lens = []
                        for c in lp:
                            lens.append(len(c))
                        print(lens)
                    print("\n")
                    #overlap = []
                    #for i in range(8):
                     #   common_set = set(C[0][i])
                      #  union_set  = set(C[0][i])
                       # for j in range(1,len(C)):
                        #    current_set = set(C[j][i])
                         #   common_set  = common_set.intersection(current_set)
                          #  union_set   = union_set.union(current_set)
                        #if len(union_set) > 0:
                         #   overlap_percentage = len(common_set) / len(union_set)
                       # else:
                        #    overlap_percentage = 0
                        #overlap.append(overlap_percentage)
                   # print(overlap)

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
                for lps in os.listdir(cat_loc):
                    act_dic = {}  
                    lp_loc = os.path.join(cat_loc,lps)
                    lp_name = lps
                    cluster_dic = {}
                    for lyr in range(n_lyrs):
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
                        print(combined_array)
                        n_clusters = 8
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        kmeans.fit(combined_array.T)
                        labels = kmeans.labels_
                        clusters = [[] for _ in range(n_clusters)]
                        X = np.arange(1, (len(labels)+1), 1, dtype=int)
                        for data_point, cluster_id in zip(X, labels):
                            clusters[cluster_id].append(str(data_point))
                        save_loc0 = os.path.join(os.getcwd(),"cluster_lists")
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
idnfy.compare()
