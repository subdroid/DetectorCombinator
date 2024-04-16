import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

filename = os.path.join(os.getcwd(),"silhoutte_scores.json")
with open(filename) as infile:
    data = json.load(infile)

max_len = -7
for clus in data.keys():
    for model in data[clus].keys():
        for cat in data[clus][model].keys():
            for lang in data[clus][model][cat].keys():
                if len(data[clus][model][cat][lang]) > max_len:
                    max_len = len(data[clus][model][cat][lang])
Data = []
for clus in data.keys():
    for model in data[clus].keys():
        for cat in data[clus][model].keys():
            for lang in data[clus][model][cat].keys():
                D = data[clus][model][cat][lang]
                if len(D) < max_len:
                    D = D + ["nil"]*(max_len-len(D))
                temp1 = [clus,model,cat,lang]
                temp2 = D 
                temp = temp1 + temp2
                Data.append(temp)
headings = ["Cluster","Model","Category","Language"]
extract = []
for i in range(max_len):
    headings.append(f"Layer_{i+1}")
    extract.append(f"Layer_{i+1}")
Data_df = pd.DataFrame(Data,columns=headings)

for cat in Data_df['Category'].unique():
    cat_data = Data_df[Data_df['Category'] == cat]
    clusters = cat_data['Cluster'].unique()
    colors = ['r','g','b','c','m','y','k','w']
    for cid, c in enumerate(clusters):
        c_data = cat_data[cat_data['Cluster'] == c]
        sel_data = c_data[extract].values.tolist()
        plot_data = []
        for row in sel_data:
            R = []
            for r in row:
                if r != "nil":
                    R.append(r)
            R = np.mean(R)
            plot_data.append(R)
        plt.plot(plot_data,color=colors[cid],label=c)
    plt.legend()
    plt.xlabel("Datasets")
    plt.ylabel("Silhoutte Score")
    plt.title(f"Silhoutte Score for {cat}")
    plt.savefig(f"{cat}.png")
    plt.clf()
    plt.close()
    # break








# print(Data_df['Cluster'].unique())
# print(Data_df.head())
# print(Data_df['Category'].unique())
# for cat in Data_df['Category'].unique():
#     cat_data = Data_df[Data_df['Category'] == cat]
#     print(cat_data.head())
    # print(Data_df[Data_df['Category'] == cat].head())
    # print(Data_df[Data_df['Category'] == cat].tail())
    # print("\n\n\n")