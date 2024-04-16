import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import json

class find_multilingual_neurons():

    def __init__(self):
        self.results = os.path.join(os.getcwd(), 'results')
    
    def extract_dictionary(self, model_loc, neuron_loc, act_type, model):
        model_neuron_loc = os.path.join(neuron_loc, model)
        if not os.path.exists(model_neuron_loc):
            os.mkdir(model_neuron_loc)
        model_neuron_loc = os.path.join(model_neuron_loc, act_type)
        if not os.path.exists(model_neuron_loc):
            os.mkdir(model_neuron_loc)        
        for langs in os.listdir(model_loc):
            lang_name = langs.split('.')[0]
            lang_loc = os.path.join(model_loc, langs)
            f_cont = pd.read_csv(lang_loc,sep=',',header=None).to_numpy()
            lang_data = {}
            for rid in range(f_cont.shape[0]):
                lang_data[str(rid)] = {}
                row = {}
                row_data = f_cont[rid]
                indices = np.arange(1,len(row_data)+1)
                for index, ind in enumerate(indices):
                    row[str(ind)] = row_data[index]
                sorted_items = sorted(row.items(), key=lambda item: item[1], reverse=True)
                sorted_row = {}
                for it in sorted_items:
                    sorted_row[it[0]]=it[1]
                lang_data[str(rid)] = sorted_row
                # print(sorted_row)
                # print(sorted_items)
                # sorted_row = []
                # for it in sorted_items:
                #     sorted_row.append(it[0])
                # break
            save_loc = os.path.join(model_neuron_loc,lang_name+".json")
            with open(save_loc,"w") as f:
                json.dump(lang_data,f)
             
    def create_dic(self):
        neuron_loc = os.path.join(os.getcwd(),"neuron_lists")
        if not os.path.exists(neuron_loc):
            os.mkdir(neuron_loc)
        for models in os.listdir(self.results):
            self.model_loc = os.path.join(self.results, models)
            detector_loc = os.path.join(self.model_loc, 'detector_act')
            self.extract_dictionary(detector_loc,neuron_loc,'detector',models)
            combinator_loc = os.path.join(self.model_loc, 'combinator_act')
            self.extract_dictionary(combinator_loc,neuron_loc,'combinator',models)
            # break

    def check_entropy(self):
        model_entropy_loc = os.path.join(os.getcwd(),"model_entropy")
        if not os.path.exists(model_entropy_loc):
            os.mkdir(model_entropy_loc)
        for models in os.listdir(self.results):
            model_loc = os.path.join(self.results, models)
            lang_name = []
            for cat in os.listdir(model_loc):
                cat_loc = os.path.join(model_loc, cat)
                entropy = {}
                entropy_cat_loc = os.path.join(model_entropy_loc,cat)
                if not os.path.exists(entropy_cat_loc):
                    os.mkdir(entropy_cat_loc)
                for lang in os.listdir(cat_loc):
                    lang_loc = os.path.join(cat_loc, lang)
                    lang_name = lang.replace('.csv','')
                    if lang not in entropy.keys():
                        entropy[lang_name] = {}
                    f_cont = pd.read_csv(lang_loc,sep=',',header=None)
                    f_cont = f_cont.to_numpy()
                    for rid in range(f_cont.shape[0]):
                        row = f_cont[rid]
                        row_max = np.max(row)
                        row = row/row_max # Activation probability
                        epsilon = 1e-10  # Small constant to avoid division by zero
                        row = row + epsilon
                        log_probs = np.where(row > 0, np.log2(row), 0)
                        ent = -np.sum(row*log_probs)
                        # ent = "{:.2e}".format(ent)
                        entropy[lang_name][rid] = ent
                entropy_df = pd.DataFrame(entropy)
                entropy_df.to_csv(os.path.join(model_entropy_loc,models+"_"+cat+".csv"))

    def plot_entropy(self):
        model_entropy_loc = os.path.join(os.getcwd(),"model_entropy")
        for files in os.listdir(model_entropy_loc):
            if ".csv" in files:
                fnames = files.replace(".csv","").split("_")
                model_name = fnames[0].replace('.','-')
                category = fnames[1]
                file_df = pd.read_csv(os.path.join(model_entropy_loc,files),index_col=0)
                langs = ["de","cs","fr","hi","Ende"]
                colors = {"de":'indianred',"cs":'salmon',
                          "fr":'royalblue',"hi":'forestgreen',
                          "Ende":'orange'}
                for i, col in enumerate(file_df.columns):
                    if col in langs:
                        c = colors[col]
                        plottable = (1/file_df[col])
                        plt.plot(plottable,linestyle='-', marker='o', label=col,color=c)
                        # plt.errorbar(plottable.index, plottable, yerr=np.std(plottable), fmt='o', color=c)
                plt.xlabel('Layer No.')
                plt.ylabel(f'Extent of Activation({category})')                        
                fol_loc = os.path.join(model_entropy_loc,category)
                if not os.path.exists(fol_loc):
                    os.mkdir(fol_loc)
                if category=='detector':
                    plot_name = f"d_all_{model_name}.pdf"
                else:
                    plot_name = f"c_all_{model_name}.pdf"
                plt.title(model_name,fontsize=20)
                plt.legend()
                plt.savefig(os.path.join(fol_loc,plot_name))
                plt.close()
                plt.clf()        

    def overlap(self,list1,list2):
        count = 0
        total = 0 
        for item in list1:
            if item in list2:
                count += 1
            total += 1
        return (count/total)*100

    def find_eng_overlap(self):
        self.data_loc = os.path.join(os.getcwd(), 'data')
        english_sents = {}
        for files in os.listdir(self.data_loc):
            suffix = files.split('.')[0]
            if len(suffix.split('_'))==2:
                lang_pair = suffix.split('_')[0]
                lang_name = suffix.split('_')[1]
                if lang_name == 'en':
                    pair = lang_pair[:2]+"_"+lang_pair[2:]
                    lines = open(os.path.join(self.data_loc, files), 'r').read().split("\n")
                    L = []
                    for lid, line in enumerate(lines):
                        if lid==1000:
                            break
                        if line.strip():
                            L.append(line)
                    english_sents[pair] = L            
        legends = list(english_sents.keys())
        pers = []
        for r in english_sents.keys():
            r_item = english_sents[r]
            P = []
            for c in english_sents.keys():
                c_item = english_sents[c]
                percentage = self.overlap(r_item,c_item)
                P.append(percentage)
            pers.append(P)
        per_array = np.array(pers)
        np.fill_diagonal(per_array, np.nan)
        ax = sns.heatmap(per_array, linewidth = 1 , cmap = 'copper', linecolor='black', annot=True, fmt=".0f")
        plt.title('Overlap of English sentences across datasets',fontdict={'fontsize': 15})
        plt.xticks(np.arange(len(legends))+0.5, legends, fontsize=8, weight='bold')
        plt.yticks(np.arange(len(legends))+0.5, legends, fontsize=8, weight='bold')
        plt.grid(which="minor", color="black", linewidth=1)
        # plt.savefig("overlap_english_sents")
        plt.savefig("overlap_english_sents.pdf")
        return None

    def compare_english(self):
        neuron_list_loc = os.path.join(os.getcwd(),"neuron_lists")
        english_files = ["Encs.json","Ende.json","Enfr.json","Enhi.json"]
        plot_loc = os.path.join(os.getcwd(),"english_overlap")
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)
        for models in os.listdir(neuron_list_loc):
            model_name = models.replace('.','_')
            model_loc = os.path.join(neuron_list_loc,models)
            for cat in os.listdir(model_loc):
                cat_loc = os.path.join(model_loc,cat)
                # freeze = [0.05,0.10,0.20,0.30,0.40,0.50]
                freeze = [0.05,0.20,0.40,0.80]
                n_subplots = len(freeze)
                n_cols = 2
                n_rows = math.ceil(n_subplots/n_cols)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(5  *n_cols, 4*n_rows)) 
                fig.subplots_adjust(hspace=0.5)  # Increase vertical space between rows
                markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
                colors  = ["salmon","darkorange","teal","royalblue"]  
                for fid, f in enumerate(freeze):
                    ax = axs[fid//2,fid%2] if n_subplots > 1 else axs
                    for i in range(len(english_files)):
                        for j in range(len(english_files)):
                            if i!=j:
                                i_tile = english_files[i].replace('.json','')
                                j_tile = english_files[j].replace('.json','')
                                f_loc1 = os.path.join(cat_loc,english_files[i])
                                df1    = json.load(open(f_loc1,"r"))
                                f_loc2 = os.path.join(cat_loc,english_files[j])
                                df2    = json.load(open(f_loc2,"r"))
                                match  = []
                                for lyr in df1.keys():
                                    D1 = list(df1[lyr].keys())
                                    D2 = list(df2[lyr].keys())
                                    l_total = len(D1) 
                                    freeze_len = int(len(D1)*f)
                                    D1 = D1[:freeze_len]
                                    D2 = D2[:freeze_len]
                                    D1 = set(D1)
                                    D2 = set(D2)
                                    intersection = D1.intersection(D2)
                                    ratio = np.around((len(intersection)/len(D1))*100,2)
                                    match.append(ratio)
                                ttl = str(f)+"_"+i_tile+"_"+j_tile
                                ax.title.set_text(f"freeze (top {int(f*100)}%): {freeze_len} neurons of {l_total}")
                                ax.plot(match, label=ttl, marker=markers[fid], color=colors[j])
                        break
                    ax.legend()
                plt.legend()
                plt.suptitle(f"Overlap of English specific neurons: {model_name} ({cat})", fontsize=22)
                fig_loc = os.path.join(plot_loc,model_name+"_"+cat  +".pdf")
                plt.tight_layout()
                plt.savefig(fig_loc)
                plt.close()
                plt.clf()

find_neurons = find_multilingual_neurons()
# find_neurons.find_eng_overlap()
# find_neurons.check_entropy()
find_neurons.plot_entropy()
# find_neurons.create_dic()
# find_neurons.compare_english()
