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
                sorted_row = dict(sorted_items)
                # print(sorted_row)
                # lang_data[str(rid)] = row
                lang_data[str(rid)] = sorted_row
            save_loc = os.path.join(model_neuron_loc,lang_name+".json")
            with open(save_loc, 'a') as file:
                json.dump(lang_data, file, indent=4)
             
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
                        row = torch.tensor(row)
                        # log_probs = torch.log2(row)
                        log_probs = torch.where(row > 0, torch.log2(row), 0)
                        ent = -torch.sum(row * log_probs)
                        ent = ent.cpu().detach().numpy()
                        ent = "{:.2e}".format(ent)
                        entropy[lang_name][rid] = ent
                entropy_df = pd.DataFrame(entropy)
                # print(os.path.join(model_entropy_loc,models+"_"+cat+".csv"))
                entropy_df.to_csv(os.path.join(model_entropy_loc,models+"_"+cat+".csv"))
                english_files = ["Enhi","Enfr","Encs","Ende"]
                # # line_styles = ['-', '--', '-.', ':']  # Different line styles
                # # markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']  # Different markers
                colors = ['indianred','salmon','royalblue','forestgreen']
                for c in english_files:
                    c_name = c.replace('En','')
                    if c_name=='de':
                        color = colors[0]
                    if c_name=='cs':
                        color = colors[1]
                    if c_name=='fr':    
                        color = colors[2] 
                    if c_name=='hi':
                        color = colors[3]
                    try:
                        data = entropy_df[c]
                        c_label = c[:2]+"("+c[2:]+")"
                        plt.plot(data,linestyle='-', marker='o',color=color,label=c_label)
                    except KeyError:
                        print(f"Caught Key Error:\t{models}\t{cat}")
                model_name = models.replace('.','-')
                plt.gca().yaxis.set_ticks([]) 
                plt.legend()
                if cat=='detector_act':
                    cat_title = "detector activations"
                if cat=='combinator_act':
                    cat_title = "combinator activations"
                plt.title(f"Entropy of {cat_title} for {models}")
                plt.ylabel('Entropy')
                plt.xlabel('Layer No.')
                plt.savefig(os.path.join(entropy_cat_loc,"english_"+model_name))
                plt.close()
                plt.clf()
                for c in entropy_df.columns:
                    if c not in english_files:
                        if c=='de':
                            color = colors[0]
                        if c=='cs':
                            color = colors[1]
                        if c=='fr':    
                            color = colors[2] 
                        if c=='hi':
                            color = colors[3]
                        data = entropy_df[c]
                        plt.plot(data,linestyle='-', marker='o',color=color,label=c)
                plt.gca().yaxis.set_ticks([]) 
                plt.legend()
                if cat=='detector_act':
                    cat_title = "detector activations"
                if cat=='combinator_act':
                    cat_title = "combinator activations"
                plt.title(f"Entropy of {cat_title} for {models}")
                plt.ylabel('Entropy')
                plt.xlabel('Layer')
                plt.savefig(os.path.join(entropy_cat_loc,"all_"+model_name))
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
        plt.savefig("overlap_english_sents")
        return None

    def sort_dic(self):
        neuron_list_loc 
        return None
find_neurons = find_multilingual_neurons()
# find_neurons.find_eng_overlap()
# find_neurons.check_entropy()
# find_neurons.create_dic()
find_neurons.compare_english()