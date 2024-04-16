import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
import torch
import gzip
import torch.nn as nn
import sys
import math
import scipy.stats as ss
from scipy import stats
import seaborn as sns
# from torch.nn.functional import pairwise_distance

class mechanistic():

    def __init__(self):
        self.results = os.path.join(os.getcwd(), 'snapshots')

        self.sparsity_plot = os.path.join(os.getcwd(), 'sparstity_plots')
        if not os.path.exists(self.sparsity_plot):
            os.mkdir(self.sparsity_plot)
        
        self.entropy_plot = os.path.join(os.getcwd(), 'entropy_plots')
        if not os.path.exists(self.entropy_plot):
            os.mkdir(self.entropy_plot)

        self.sparsity_data = os.path.join(os.getcwd(), 'sparstity_data')
        if not os.path.exists(self.sparsity_data):
            os.mkdir(self.sparsity_data)
        
        self.entropy_data = os.path.join(os.getcwd(), 'entropy_data')
        if not os.path.exists(self.entropy_data):
            os.mkdir(self.entropy_data)

        self.rank_data = os.path.join(os.getcwd(), 'rank_data')
        if not os.path.exists(self.rank_data):
            os.mkdir(self.rank_data)

    def activation(self,array):
        with gzip.open(array, 'rb') as f:
            data = np.load(f)
            data = torch.tensor(data,dtype=torch.float32).cuda()
            m = nn.GELU()
            activation = m(data)
            num_zeros = torch.eq(activation, 0.0).sum().item()
            ratio_zeros = (num_zeros/activation.numel())*100
            activation_frequency = torch.sum(activation != 0, dim=0)/activation.shape[0]
            activation_frequency = activation_frequency.cpu().numpy()
 
            m2 = nn.ReLU()
            activation2 = m2(data)
            num_zeros2 = torch.eq(activation2, 0.0).sum().item()
            ratio_zeros2 = (num_zeros2/activation2.numel())*100
            activation_frequency2 = torch.sum(activation2 != 0, dim=0)/activation2.shape[0]
            activation_frequency2 = activation_frequency2.cpu().numpy()

            return ratio_zeros, ratio_zeros2, activation_frequency.tolist(), activation_frequency2.tolist()
            
    def extract_sparsity(self):
        models = os.listdir(self.results)
        model = models[int(sys.argv[1])]
        model_loc = os.path.join(self.results, model)
        for cat in os.listdir(model_loc):
            cat_loc = os.path.join(model_loc, cat)
            Lang_dict_zero_gelu = {}
            Lang_dict_zero_relu = {}
            Lang_dict_freq_gelu = {}
            Lang_dict_freq_relu = {}
            for lang in os.listdir(cat_loc):
                Lang_dict_zero_gelu[lang] = {}
                Lang_dict_zero_relu[lang] = {}
                Lang_dict_freq_gelu[lang] = {}
                Lang_dict_freq_relu[lang] = {}
                lang_loc = os.path.join(cat_loc, lang)
                for layer in tqdm(os.listdir(lang_loc)):
                    layer_loc = os.path.join(lang_loc, layer)
                    z1, z2, freq1, freq2  = self.activation(layer_loc)
                    Lang_dict_zero_gelu[lang][layer] = z1
                    Lang_dict_zero_relu[lang][layer] = z2
                    Lang_dict_freq_gelu[lang][layer] = freq1
                    Lang_dict_freq_relu[lang][layer] = freq2
            file_name = os.path.join(self.sparsity_data, f'{model}_{cat}_GELU_sparsity.json')
            with open(file_name, "w") as outfile: 
                json.dump(Lang_dict_zero_gelu, outfile)          
            file_name = os.path.join(self.sparsity_data, f'{model}_{cat}_RELU_sparsity.json')
            with open(file_name, "w") as outfile: 
                json.dump(Lang_dict_zero_relu, outfile)          
            file_name = os.path.join(self.sparsity_data, f'{model}_{cat}_GELU_frequency.json')
            with open(file_name, "w") as outfile: 
                json.dump(Lang_dict_freq_gelu, outfile)
            file_name = os.path.join(self.sparsity_data, f'{model}_{cat}_RELU_frequency.json')
            with open(file_name, "w") as outfile: 
                json.dump(Lang_dict_freq_relu, outfile)        

    def plotter_sparsity(self,data,model,cat,act,type_data,num_lyr,colors,intra):        
        # print("sparsity")
        sparsity_loc = os.path.join(self.sparsity_plot, 'Sparsity')
        if not os.path.exists(sparsity_loc):
            os.mkdir(sparsity_loc)
        plt.figure(figsize=(5, 3))  # Adjust the size as needed
        for lang in sorted(data.keys()):
            plot_flag = False
            if intra=="inter":
                if lang!="Encs" and lang!="Enhi" and lang!="Enfr":
                    plot_flag = True
            elif intra=="intra":
                plot_flag = False
                # if "En" in lang:
                #     plot_flag = True
            if act=="RELU":
                plot_flag=False
            if plot_flag:
                sparsity_data = data[lang]
                L_data = []
                for l in range(num_lyr):
                    lyr = f"{l}.npy.gz" 
                    if lyr in sparsity_data.keys():
                        L_data.append(sparsity_data[lyr])
                    else:
                        L_data.append(0)
                X = np.arange(0,len(L_data))
                label = lang
                if intra=="inter":
                    if "En" in lang:
                        label = "en"
                plt.plot(X,L_data, label=label, color=colors[lang], linestyle='dotted', marker='8')
        
        if intra=="inter" and act=="GELU":
            # plt.title(f'Inter-language sparsity ({act}): {model}--> {cat}')
            plt.title(f'{model}',fontsize=20)
            plt.xlabel('Layers')
            plt.ylabel('Sparsity')
            # plt.legend()
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(sparsity_loc, f'sparse_{intra}_{model}_{cat}_{act}.pdf'))
        plt.close()
        
    def plotter_freq(self,data,model,cat,act,type_data,num_lyr,colors,intra):   
        # print("frequency")
        sparsity_loc = os.path.join(self.sparsity_plot, 'Frequency')
        if not os.path.exists(sparsity_loc):
            os.mkdir(sparsity_loc)
        # Adjust figure size
        plt.figure(figsize=(5, 3))  # Adjust the size as needed
        for lang in sorted(data.keys()):
            plot_flag = False
            if intra=="inter":
                if lang!="Encs" and lang!="Enhi" and lang!="Enfr":
                    plot_flag = True
            elif intra=="intra":
                plot_flag = False
                # if "En" in lang:
                #     plot_flag = True
            if act=="RELU":
                plot_flag=False
            if plot_flag:
                if lang=="Ende" or lang=="hi" or lang=="de":
                    freq_data = data[lang]
                    means = []
                    std = []
                    for l in range(num_lyr):
                        lyr = f"{l}.npy.gz" 
                        if lyr in freq_data.keys():
                            req_data = freq_data[lyr]
                        else:
                            req_data = 0                    
                        means.append(np.mean(req_data))
                        std.append(np.std(req_data))
                    label = lang
                    if lang=="Ende":
                        label = "en"
                    plt.errorbar(np.arange(0,len(means)),means, yerr=std, label=label, color=colors[lang], 
                                ecolor=colors[lang], elinewidth=0.5, capsize=3, capthick=1.5, 
                                linestyle='dotted', marker='8', markersize=4)
        if intra=="inter" and act=="GELU":
            # plt.title(f'Inter-language sparsity ({act}): {model}--> {cat}')
            plt.title(f'{model}',fontsize=20)
            plt.xlabel('Layers')
            plt.ylabel('Sparsity')
            # plt.legend()
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(sparsity_loc, f'freq_{intra}_{model}_{cat}_{act}.pdf'))
        plt.close()

    def plot_sparsity(self):
        nlayers  = {"xglm-564M":24,"xglm-1.7B":24,"xglm-2.9B":48,"xglm-4.5B":4824,"xglm-7.5B":32}
        for cat_file in os.listdir(self.sparsity_data):
            name_split = cat_file.split('_')
            model = name_split[0]
            cat = name_split[1]
            act = name_split[2]
            type_data = name_split[3].replace('.json','')
            cat_loc = os.path.join(self.sparsity_data, cat_file)
            data = json.load(open(cat_loc))
            num_lyr = nlayers[model]
            colors = {
                     "de":   "forestgreen",
                     "cs":   "black",
                     "hi":   "brown",
                     "fr":   "orange",
                     "Ende": "navy",
                     "Enfr": "darkblue",
                     "Enhi": "slateblue",
                     "Encs": "cornflowerblue",
                    }    
            if type_data=="sparsity":
                c=1
                self.plotter_sparsity(data,model,cat,act,type_data,num_lyr,colors,"inter")
                self.plotter_sparsity(data,model,cat,act,type_data,num_lyr,colors,"intra")                
        
            elif type_data=="frequency":
                self.plotter_freq(data,model,cat,act,type_data,num_lyr,colors,"inter")
                self.plotter_freq(data,model,cat,act,type_data,num_lyr,colors,"intra")

    def entropy(self, data):
        min_data = torch.min(data)
        data = data - min_data
        if torch.min(data)<0:
            print("Negative values in data")
        denominator = torch.max(data)-min_data
        modified_data = data/denominator
        modified_data = modified_data + 1e-32
        info = -torch.sum(modified_data * torch.log2(modified_data))/modified_data.shape[0]
        return info

    def activation_entropy(self,data):
        with gzip.open(data, 'rb') as f:
            ndata = np.load(f)
            m = nn.GELU()
            tensor = torch.tensor(ndata,dtype=torch.float32).cuda()
            activation = m(tensor)
            results = torch.stack([self.entropy(row) for row in activation])
            results = torch.sum(results)
            results = results.cpu().numpy()
            
            return results
        
    def entropy_calculation(self):
        models = os.listdir(self.results)
        model = models[int(sys.argv[1])]
        model_loc = os.path.join(self.results, model)
        for cat in os.listdir(model_loc):
            cat_loc = os.path.join(model_loc, cat)
            Lang_dict_entropy = {}
            for lang in os.listdir(cat_loc):
                Lang_dict_entropy[lang] = {}
                lang_loc = os.path.join(cat_loc, lang)
                fcount = 0
                n_layers = len(os.listdir(lang_loc))
                for layer in tqdm(range(n_layers)):
                    layer_loc = os.path.join(lang_loc, f"{layer}.npy.gz")
                    flatness  = self.activation_entropy(layer_loc)
                    Lang_dict_entropy[lang][layer] = flatness.tolist()
            file_name = os.path.join(self.entropy_data, f'{model}_{cat}_entropy.json')
            with open(file_name, "w") as outfile: 
                json.dump(Lang_dict_entropy, outfile)

    def plot_entropy(self):
        # colors = {
        #              "de":   "lightpink",
        #              "cs":   "limegreen",
        #              "fr":   "cyan",
        #              "hi":   "forestgreen",
        #              "Ende": "navy",
        #              "Enfr": "darkblue",
        #              "Enhi": "slateblue",
        #              "Encs": "cornflowerblue",
        #             }
        colors = {
                     "de":   "forestgreen",
                     "cs":   "black",
                     "hi":   "brown",
                     "fr":   "orange",
                     "Ende": "navy",
                     "Enfr": "darkblue",
                     "Enhi": "slateblue",
                     "Encs": "cornflowerblue",
                    }
        for file in os.listdir(self.entropy_data):
            c = 0
            name_split = file.split('_')
            model = name_split[0]
            cat = name_split[1]
            cat_loc = os.path.join(self.entropy_data, file)
            # print(file)
            data = json.load(open(cat_loc))
            # plt.figure(figsize=(8, 9))  # Adjust the size as needed
            for lang in sorted(data.keys()):
                clr = colors[lang]
                if lang!="Encs" and lang!="Enhi" and lang!="Enfr":
                    L = []
                    for lid in range(len(data[lang].keys())):
                        L.append(data[lang][str(lid)])
                    if lang=="Ende":
                        lang="en"
                    plt.plot(np.arange(0,len(L)),L, label=lang, color=clr, linestyle='dotted', marker='8')
            plt.title(f'{model}',fontsize=22)
            plt.xlabel('Layers')
            plt.ylabel('Activation Flatness')
            # plt.legend()
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=17)
            plt.tight_layout()
            plt.ylim(0, 8500)
            plt.yticks(np.arange(0, 8501, 1000))
            plt.xlim(0,48)
            plt.xticks(np.arange(0, 49, 3))
            plt.savefig(os.path.join(self.entropy_plot, f'{model}_{cat}.pdf'))
            plt.close()
        
    def neuron_identify(self,array):
        with gzip.open(array, 'rb') as f:
            data = np.load(f)
            data = torch.tensor(data,dtype=torch.float32).cuda()
            m = nn.GELU()
            activation = m(data)
            sum_activation = torch.sum(activation,dim=0)
            sum_activation = sum_activation.cpu().numpy()
            ranks = ss.rankdata(sum_activation)
            return ranks.astype(int)
            
    def extract_neurons(self):
        models = os.listdir(self.results)
        model = models[int(sys.argv[1])]
        model_loc = os.path.join(self.results, model)
        for cat in os.listdir(model_loc):
            cat_loc = os.path.join(model_loc, cat)
            ranks = {}
            for lang in os.listdir(cat_loc):
                lang_loc = os.path.join(cat_loc, lang)
                ranks[lang]={}
                for layer in tqdm(os.listdir(lang_loc)):
                    layer_loc = os.path.join(lang_loc, layer)
                    rank_lyr = self.neuron_identify(layer_loc)
                    ranks[lang][layer] = rank_lyr.tolist() 
            file_name = os.path.join(self.rank_data, f'{model}_{cat}_rank.json')
            with open(file_name, "w") as outfile: 
                json.dump(ranks, outfile)                
    
    def rank_corr(self):
        colors = {
                     "de":   "lightpink",
                     "cs":   "limegreen",
                     "fr":   "cyan",
                     "hi":   "forestgreen",
                     "Ende": "navy",
                     "Enfr": "darkblue",
                     "Enhi": "slateblue",
                     "Encs": "cornflowerblue",
                    }
        markers = {
                     "de":   ".",
                     "cs":   ",",
                     "fr":   "o",
                     "hi":   "v",
                     "Ende": "^",
                     "Enfr": "*",
                     "Enhi": "+",
                     "Encs": "x",
                    }
        for files in os.listdir(self.rank_data):
            file_loc = os.path.join(self.rank_data, files)
            name_split = files.split("_")
            f_name = f"{name_split[0]}_{name_split[1]}.png"
            with open(file_loc, "r") as outfile: 
                d = json.load(outfile)  
                Rank_mat = []
                pairs = []
                sorted_keys = sorted(d.keys())
                if len(sorted_keys)>=2:
                    for k1 in sorted_keys:
                        for k2 in sorted_keys:
                            if k1!=k2:
                                if (k1 not in ["Encs","Enfr","Enhi"]) and (k2 not in ["Encs","Enfr","Enhi"]):
                                    lang1 = d[k1]
                                    lang2 = d[k2]
                                    rank_arr = []
                                    for lyr in range(len(lang1.keys())):
                                        lyr_name = f"{lyr}.npy.gz"
                                        lang1_lyr = lang1[lyr_name]
                                        lang2_lyr = lang2[lyr_name]
                                        res = stats.spearmanr(lang1_lyr,lang2_lyr)
                                        stat, pval = res[0], res[1]
                                        rank_arr.append(stat)
                                    l1 = k1
                                    l2 = k2
                                    if k1=="Ende":
                                        l1 = "en"
                                    elif k2=="Ende":
                                        l2 = "en"
                                    lyr_ttl = f"{l1}-{l2}"
                                    rev_lyr_ttl = f"{l2}-{l1}"
                                    if lyr_ttl not in pairs and rev_lyr_ttl not in pairs:
                                        pairs.append(lyr_ttl)
                                        r = [lyr_ttl] + rank_arr
                                        Rank_mat.append(r)
                rank_df = pd.DataFrame(Rank_mat)
                rank_df.set_index(rank_df.columns[0], inplace=True)
                sns.heatmap(rank_df, cmap='coolwarm')
                plt.title(f"{name_split[0]}_{name_split[1]}")
                plt.xlabel('Layers')
                plt.ylabel('Language Pairs')
                plt.tight_layout()
                plot_fol_loc = os.path.join(os.getcwd(),"rank_plots")
                if not os.path.exists(plot_fol_loc):
                    os.mkdir(plot_fol_loc)
                plot_fol_sv = os.path.join(plot_fol_loc,"inter_lang")
                if not os.path.exists(plot_fol_sv):
                    os.mkdir(plot_fol_sv)
                plot_loc = os.path.join(plot_fol_sv,f_name)
                plt.savefig(plot_loc)
                plt.close()
                plt.clf()

                Rank_mat = []
                pairs = []
                if len(sorted_keys)>=2:
                    for k1 in sorted_keys:
                        for k2 in sorted_keys:
                            if k1!=k2:
                                if ("En" in k1) and ("En" in k2):
                                    lang1 = d[k1]
                                    lang2 = d[k2]
                                    rank_arr = []
                                    for lyr in range(len(lang1.keys())):
                                        lyr_name = f"{lyr}.npy.gz"
                                        lang1_lyr = lang1[lyr_name]
                                        lang2_lyr = lang2[lyr_name]
                                        res = stats.spearmanr(lang1_lyr,lang2_lyr)
                                        stat, pval = res[0], res[1]
                                        rank_arr.append(stat)
                                    l1 = k1
                                    l2 = k2
                                    lyr_ttl = f"{l1}-{l2}"
                                    rev_lyr_ttl = f"{l2}-{l1}"
                                    if lyr_ttl not in pairs and rev_lyr_ttl not in pairs:
                                        pairs.append(lyr_ttl)
                                        r = [lyr_ttl] + rank_arr
                                        Rank_mat.append(r)
                rank_df = pd.DataFrame(Rank_mat)
                rank_df.set_index(rank_df.columns[0], inplace=True)
                sns.heatmap(rank_df, cmap='coolwarm')
                plt.title(f"{name_split[0]}_{name_split[1]}")
                plt.xlabel('Layers')
                plt.ylabel('Language Pairs')
                plt.tight_layout()
                plot_fol_sv = os.path.join(plot_fol_loc,"intra_lang")
                if not os.path.exists(plot_fol_sv):
                    os.mkdir(plot_fol_sv)
                plot_loc = os.path.join(plot_fol_sv,f"eng_{f_name}")
                plt.savefig(plot_loc)
                plt.close()
                plt.clf()

                Rank_mat = []
                pairs = []
                if len(sorted_keys)>=2:
                    for k1 in sorted_keys:
                        for k2 in sorted_keys:
                            if k1!=k2:
                                if ("En" in k2):
                                    if k2.replace("En","")==k1:
                                        lang1 = d[k1]
                                        lang2 = d[k2]
                                        rank_arr = []
                                        for lyr in range(len(lang1.keys())):
                                            lyr_name = f"{lyr}.npy.gz"
                                            lang1_lyr = lang1[lyr_name]
                                            lang2_lyr = lang2[lyr_name]
                                            res = stats.spearmanr(lang1_lyr,lang2_lyr)
                                            stat, pval = res[0], res[1]
                                            rank_arr.append(stat)
                                        l1 = k1
                                        l2 = k2
                                        lyr_ttl = f"{l1}-{l2}"
                                        rev_lyr_ttl = f"{l2}-{l1}"
                                        if lyr_ttl not in pairs and rev_lyr_ttl not in pairs:
                                            pairs.append(lyr_ttl)
                                            r = [lyr_ttl] + rank_arr
                                            Rank_mat.append(r)
                rank_df = pd.DataFrame(Rank_mat)
                rank_df.set_index(rank_df.columns[0], inplace=True)
                sns.heatmap(rank_df, cmap='coolwarm')
                plt.title(f"{name_split[0]}_{name_split[1]}")
                plt.xlabel('Layers')
                plt.ylabel('Language Pairs')
                plt.tight_layout()
                plot_fol_sv = os.path.join(plot_fol_loc,"parallel")
                if not os.path.exists(plot_fol_sv):
                    os.mkdir(plot_fol_sv)
                plot_loc = os.path.join(plot_fol_sv,f"parallel_{f_name}")
                plt.savefig(plot_loc)
                plt.close()
                plt.clf()

    def investigate_flatness_comb(self):
        def normalize(data):
            min_data = torch.min(data)
            data = data - min_data
            if torch.min(data)<0:
                print("Negative values in data")
            denominator = torch.max(data)-min_data
            modified_data = data/denominator
            modified_data = modified_data + 1e-32
            return modified_data

        combinator_snaps = os.path.join(os.getcwd(), 'combinator_pics')
        if not os.path.exists(combinator_snaps):
            os.mkdir(combinator_snaps)
        for model in os.listdir(self.results):
            if model=="xglm-2.9B" or model=="xglm-7.5B":                
                model_pics = os.path.join(combinator_snaps, model)
                if not os.path.exists(model_pics):
                    os.mkdir(model_pics)
                model_loc = os.path.join(self.results, model)
                comb_loc  = os.path.join(model_loc,"combinators")
                end_loc = os.path.join(comb_loc,"Ende")
                n_layers = len(os.listdir(end_loc))
                lang_list = ["Ende","Encs","de","cs"]
                # colors = {"Ende":"firebrick","Enfr":"cornflowerblue","de":"lightcoral","fr":"dodgerblue"}
                colors = {
                        "de":   "forestgreen",
                        "cs":   "black",
                        "hi":   "brown",
                        "fr":   "orange",
                        "Ende": "navy",
                        "Enfr": "darkblue",
                        "Enhi": "slateblue",
                        "Encs": "cornflowerblue",
                    }
                for lyr_id in range(n_layers):
                    print(model,lyr_id)
                    lyr_name = f"{lyr_id}.npy.gz"
                    file_name = os.path.join(model_pics, f'{lyr_id}.png')
                    plt.figure(figsize=(3, 3))  # Adjust the size as needed
                    fig, axs = plt.subplots(2,2)
                    for lid, lang in enumerate(lang_list):
                        lang_loc = os.path.join(comb_loc, lang)
                        lyr_loc = os.path.join(lang_loc, lyr_name)
                        print(lyr_loc)
                        with gzip.open(lyr_loc, 'rb') as f:
                            data = np.load(f)
                            data = torch.tensor(data,dtype=torch.float32).cuda()
                            m = nn.GELU()
                            activation = m(data)
                            results = torch.stack([normalize(row) for row in activation])
                            results = results.cpu().numpy()    
                            for i in tqdm(range(results.shape[1])):
                                axs[int(lid/2),int(lid%2)].plot(np.repeat(i,len(results[:,i])),results[:, i],color=colors[lang])
                            ttl = f"{lang}"
                            axs[int(lid/2),lid%2].set_title(ttl,fontsize=15)
                    print(file_name)    
                    plt.suptitle(f'{model}: {lyr_id}',fontsize=20)
                    plt.tight_layout()
                    plt.savefig(file_name)
                    plt.close()
                    plt.clf()
                    
    def investigate_flatness_det(self):
        def normalize(data):
            min_data = torch.min(data)
            data = data - min_data
            if torch.min(data)<0:
                print("Negative values in data")
            denominator = torch.max(data)-min_data
            modified_data = data/denominator
            modified_data = modified_data + 1e-32
            return modified_data

        detector_snaps = os.path.join(os.getcwd(), 'detector_pics')
        if not os.path.exists(detector_snaps):
            os.mkdir(detector_snaps)
        for model in os.listdir(self.results):
            if model=="xglm-2.9B" or model=="xglm-7.5B":
                model_pics = os.path.join(detector_snaps, model)
                if not os.path.exists(model_pics):
                    os.mkdir(model_pics)
                model_loc = os.path.join(self.results, model)
                det_loc  = os.path.join(model_loc,"detectors")
                end_loc = os.path.join(det_loc,"Ende")
                n_layers = len(os.listdir(end_loc))
                lang_list = ["Ende","Encs","de","cs"]
                colors = {
                        "de":   "forestgreen",
                        "cs":   "black",
                        "hi":   "brown",
                        "fr":   "orange",
                        "Ende": "navy",
                        "Enfr": "darkblue",
                        "Enhi": "slateblue",
                        "Encs": "cornflowerblue",
                    }
                
                for lyr_id in range(n_layers):
                    print(model,lyr_id)
                    lyr_name = f"{lyr_id}.npy.gz"
                    file_name = os.path.join(model_pics, f'{lyr_id}.png')
                    plt.figure(figsize=(3, 3))  # Adjust the size as needed
                    fig, axs = plt.subplots(2,2)
                    for lid, lang in enumerate(lang_list):
                        lang_loc = os.path.join(det_loc, lang)
                        lyr_loc = os.path.join(lang_loc, lyr_name)
                        print(lyr_loc)
                        with gzip.open(lyr_loc, 'rb') as f:
                            data = np.load(f)
                            data = torch.tensor(data,dtype=torch.float32).cuda()
                            m = nn.GELU()
                            activation = m(data)
                            results = torch.stack([normalize(row) for row in activation])
                            results = results.cpu().numpy()
                            if lang=="cs":    
                                # print(results)
                                I = []
                                for i in tqdm(range(results.shape[1])):
                                    # print(np.max(results[:, i]),np.min(results[:, i]))
                                    if np.max(results[:, i])>0.9:
                                        I.append(i)
                                        print(results[:, i])
                                        print(np.repeat(i,len(results[:,i])))
                                    plt.plot(np.repeat(i,len(results[:,i])),results[:, i],color=colors[lang])
                                plt.savefig("Demo")
                                print(I)
                                break
                    break
                    #         for i in tqdm(range(results.shape[1])):
                    #             axs[int(lid/2),int(lid%2)].plot(np.repeat(i,len(results[:,i])),results[:, i],color=colors[lang])
                    #         ttl = f"{lang}"
                    #         axs[int(lid/2),lid%2].set_title(ttl,fontsize=10)
                        # break
                    # print(file_name)    
                    # plt.suptitle(f'{model}: {lyr_id}',fontsize=10)
                    # plt.tight_layout()
                    # plt.savefig(file_name)
                    # plt.close()
                    # plt.clf()

    def tensor_dist(self):
        def normalize(data):
            min_data = torch.min(data)
            data = data - min_data
            if torch.min(data)<0:
                print("Negative values in data")
            denominator = torch.max(data)-min_data
            modified_data = data/denominator
            modified_data = modified_data + 1e-32
            return modified_data
        cosi = torch.nn.CosineSimilarity(dim=2)
        for model in os.listdir(self.results):
            if model!="xglm-4.5B":
                model_loc = os.path.join(self.results, model)
                for cat in os.listdir(model_loc):
                    # if cat=="combinators":
                    if cat=="detectors":
                        cat_loc  = os.path.join(model_loc,cat)
                        end_loc = os.path.join(cat_loc,"Ende")
                        n_layers = len(os.listdir(end_loc))
                        lang_list = ["Ende","Enfr","de","fr"]
                        colors = {"Ende":"firebrick","Enfr":"cornflowerblue","de":"lightcoral","fr":"dodgerblue"}
                        distances = {}
                        for lyr_id in tqdm(range(n_layers)):
                            print(model,lyr_id)
                            # if lyr_id==5:
                            #     break
                            lyr_name = f"{lyr_id}.npy.gz"
                            tensors = []
                            langs = []
                            for lid, lang in enumerate(os.listdir(cat_loc)):
                                lang_loc = os.path.join(cat_loc, lang)
                                lyr_loc = os.path.join(lang_loc, lyr_name)
                                with gzip.open(lyr_loc, 'rb') as f:
                                    data = np.load(f)
                                    data = torch.tensor(data,dtype=torch.float32).cuda()
                                    m = nn.GELU()
                                    activation = m(data)
                                    results = torch.stack([normalize(row) for row in activation])
                                    tensors.append(results)
                                    # tensors.append(activation)
                                    langs.append(lang)
                            pairs = []
                            for i in range(len(tensors)):
                                for j in range(i+1, len(tensors)):
                                    pair_name = f"{langs[i]}-{langs[j]}"
                                    rev_name  = f"{langs[j]}-{langs[i]}"
                                    if "En" in langs[i] and "En" in langs[j]:
                                        break
                                    if (pair_name not in pairs) and (rev_name not in pairs): 
                                        pairs.append(pair_name)
                                        tens1 = tensors[i]
                                        tens2 = tensors[j]
                                        min_samples = min(tens1.shape[0],tens2.shape[0])
                                        index1 = torch.randint(low=0, high=min_samples, size=(1,min_samples))
                                        index2 = torch.randint(low=0, high=min_samples, size=(1,min_samples))
                                        tensor1 = tens1[index1][0]
                                        tensor2 = tens2[index2][0]
                                        dist = []
                                        for t_ind in tensor1:
                                            t1 = t_ind
                                            t2 = tensor2
                                            t1t = t1.view(t1.shape[0],1)
                                            r = torch.matmul(t1t.t(),t2.t())
                                            dist.append(torch.min(r))
                                        D = torch.sum(torch.stack(dist)).cpu().item()
                                        if not pair_name in distances.keys():
                                            distances[pair_name] = {}
                                        if not lyr_id in distances[pair_name].keys():
                                            distances[pair_name][lyr_id] = []   
                                        # distances[pair_name][lyr_id]=dist.tolist()
                                        distances[pair_name][lyr_id]=D
                        for lp in distances.keys():
                            lang_pair = distances[lp]
                            plot_data = []
                            for lyr in lang_pair.keys():
                                lyr_info = lang_pair[lyr]
                                plot_data.append(lyr_info)
                            lang1 = lp.split("-")[0]
                            lang2 = lp.split("-")[1]
                            if lang1=="de":
                                color = "black"
                            elif lang1=="fr":
                                color = "royalblue"
                            elif lang1=="hi":
                                color = "brown"
                            elif lang1=="cs":
                                color = "teal"
                            if "En" in lang1:
                                color = "springgreen"
                            if lang2=="de":
                                marker = "."
                            elif lang2=="fr":
                                marker = "v"
                            elif lang2=="hi":
                                marker = "^"
                            elif lang2=="cs":
                                marker = "4"
                            elif "En" in lang2:
                                marker = "+"
                            x = []
                            for i in range(len(plot_data)):
                                x.append(i+1)
                            plt.plot(x,plot_data,label=lp,color=color,marker=marker)
                        plt.legend(loc='center left',  fontsize="4", bbox_to_anchor=(1, 0.5), ncol=3, fancybox=True)
                        plt.xlabel("Layers")
                        plt.ylabel("Distance")
                        plt.title(f"Representational sim: {model}-->{cat}")
                        plt.tight_layout()
                        f_name = f"{model}_{cat}_distances.png"
                        f_loc  = os.path.join(os.getcwd(),"model_layer_dist")
                        if not os.path.exists(f_loc):
                            os.mkdir(f_loc)
                        sv_loc = os.path.join(f_loc,f_name)
                        plt.savefig(sv_loc)
                        plt.close()
                        plt.clf()
   
if __name__ == "__main__":
    m = mechanistic()
    # m.extract_sparsity()
    # m.plot_sparsity()
    # m.entropy_calculation()
    # m.plot_entropy()
    # m.extract_neurons()
    # m.rank_corr()
    # m.investigate_flatness_comb()
    m.investigate_flatness_det()
    # m.tensor_dist()
    