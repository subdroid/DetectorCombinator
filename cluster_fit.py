import os
import sklearn as sk
import json 
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import numpy as np
import json

def cluster_silhoutte(cluster_list):
    flattened_data = np.array([int(item) for sublist in cluster_list for item in sublist])
    labels = np.zeros(flattened_data.shape[0], dtype=int)
    for i, cluster in enumerate(cluster_list):
        for index in cluster:
            labels[int(index)-1] = i
    score = silhouette_score(flattened_data.reshape(-1,1),labels, metric='euclidean')
    return score
    

cluster_sizes = [8,16,32,64,128,256,512]
folder_prefix = "cluster_lists_"
curr_loc = os.getcwd()
C = {}
for size in cluster_sizes:
    size_id = size
    if size_id not in C.keys():
        C[size_id] = {}
    req_name = f"{folder_prefix}{str(size)}"
    req_fol = os.path.join(curr_loc,req_name)
    for model in os.listdir(req_fol):
        model_loc = os.path.join(req_fol,model)
        model_name = model
        if model_name not in C[size_id].keys():
            C[size_id][model_name] = {}
        for category in os.listdir(model_loc):
            cat_name = category
            if cat_name not in C[size_id][model_name].keys():
                C[size_id][model_name][cat_name] = {}
            category_loc = os.path.join(model_loc,category)
            for lang in os.listdir(category_loc):
                lang_name = lang
                lang_loc = os.path.join(category_loc,lang)
                print(f"{size_id} {model_name} {cat_name} {lang_name}")
                c = []
                for layers in os.listdir(lang_loc):
                    sil_scores = []
                    lyr_loc = os.path.join(lang_loc,layers)
                    with open(lyr_loc) as f:
                        data   = json.load(f)
                        cscore = cluster_silhoutte(data)
                        c.append(cscore)
                C[size_id][model_name][cat_name][lang_name] = c
    #             break
    #         break
    #     break
    # break
    
filename = os.path.join(os.getcwd(),"silhoutte_scores.json")
with open(filename, "w") as json_file:
    json.dump(C, json_file, indent=4)
# with open(filename) as infile:
#     data = json.load(infile)
#     print(data)