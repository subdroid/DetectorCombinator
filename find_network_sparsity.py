import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import json

class prober():

    def __init__(self):
        self.results = os.path.join(os.getcwd(), 'results')

    def check_variance(self):
        model_entropy_loc = os.path.join(os.getcwd(),"model_entropy")
        if not os.path.exists(model_entropy_loc):
            os.mkdir(model_entropy_loc)
        for models in os.listdir(self.results):
            model_loc = os.path.join(self.results, models)
            lang_name = []
            for cat in os.listdir(model_loc):
                cat_loc = os.path.join(model_loc, cat)
                entropy_cat_loc = os.path.join(model_entropy_loc,cat)
                if not os.path.exists(entropy_cat_loc):
                    os.mkdir(entropy_cat_loc)
                for lang in os.listdir(cat_loc):
                    flag = False
                    lang_loc = os.path.join(cat_loc, lang)
                    lang_name = lang.replace('.csv','')
                    if "En" in lang_name:
                        if lang_name=="Ende":
                            flag=True
                    else:
                        flag=True
                    if flag:
                        f_cont = pd.read_csv(lang_loc,sep=',',header=None)
                        f_cont = f_cont.to_numpy()
                        Std = []
                        for rid in range(f_cont.shape[0]):
                            row = f_cont[rid]
                            row_max = np.max(row)
                            row = row/row_max # Activation probability
                            plottable = np.std(row)
                            Std.append(plottable)
                        plt.plot(Std,label=lang_name)
                plt.xlabel('Layer No.')
                plt.ylabel(f'Std of extent of Activation({cat})')                        
                fol_loc = os.path.join(model_entropy_loc,cat)
                if cat=='detector_act':
                    plot_name = f"d_var_{models}.pdf"
                else:
                    plot_name = f"c_var_{models}.pdf"
                plt.title(models,fontsize=20)
                plt.legend()
                plt.savefig(os.path.join(fol_loc,plot_name))
                plt.close()
                plt.clf()


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
                        plottable = file_df[col]
                        plt.plot(plottable,linestyle='-', marker='o', label=col,color=c)
                        # plt.errorbar(plottable.index, plottable, yerr=np.std(plottable), fmt='o', color=c)
                plt.xlabel('Layer No.')
                plt.ylabel(f'Extent of Activation({category})')                        
                fol_loc = os.path.join(model_entropy_loc,category)
                if not os.path.exists(fol_loc):
                    os.mkdir(fol_loc)
                if category=='detector':
                    plot_name = f"d_{model_name}.pdf"
                else:
                    plot_name = f"c_{model_name}.pdf"
                plt.title(model_name,fontsize=20)
                plt.legend()
                plt.savefig(os.path.join(fol_loc,plot_name))
                plt.close()
                plt.clf()
    
    def sparsity_analysis(self):
        activations = {}
        for models in os.listdir(self.results):
            model_name = models.replace('.','-')
            model_loc = os.path.join(self.results, models)
            for cat in os.listdir(model_loc):
                cat_loc = os.path.join(model_loc, cat)
                if cat not in activations.keys():
                    activations[cat] = {}
                if models not in activations[cat].keys():
                    activations[cat][model_name] = {}
                for lang in os.listdir(cat_loc):
                    lang_loc = os.path.join(cat_loc, lang)
                    f_cont = pd.read_csv(lang_loc,sep=',',header=None)
                    f_cont = f_cont.to_numpy()
                    lang_name = lang.replace('.csv','')
                    if lang_name not in activations.keys():
                        activations[cat][model_name][lang_name] = {}
                    for rid in range(f_cont.shape[0]):
                        lyr_act = f_cont[rid]
                        lyr_max = np.max(lyr_act)
                        lyr_norm = lyr_act/lyr_max
                        activations[cat][model_name][lang_name][str(rid)] = lyr_norm
        
        for cat in activations.keys():
            cat_data = activations[cat]
            model_names = list(cat_data.keys())
            lang_names  = list(cat_data['xglm-1-7B'].keys())
            lang_pairs = []
            for lang in lang_names:
                if "En" in lang:
                    l2 = lang[2:]
                    lang_pairs.append([lang,l2])
            # print(lang_pairs)
            layer_plots = os.path.join(os.getcwd(),"layer_activation_plots")
            for model in model_names:
                model_plot_loc = os.path.join(layer_plots,model)
                if not os.path.exists(model_plot_loc):
                    os.makedirs(model_plot_loc)
            #     lang_names  = list(cat_data[model].keys())
                n_lyrs = cat_data[model][lang_names[0]].keys()
                for lyr in n_lyrs:
                    for pairs in lang_pairs:
                        p_name = pairs[0]
                        plot_name = lyr + "_" + p_name + ".png"
                        plot_loc = os.path.join(model_plot_loc,plot_name)
                        for lid, lang in enumerate(pairs):
                            model_data = cat_data[model][lang][lyr]
                            if lid==0:
                                m='b'
                            elif lid==1:
                                m='r'
                            plt.plot(model_data,m,label=lang)
                        plt.legend()
                        plt.savefig(plot_loc)
                        plt.close()
                        plt.clf()

probe = prober()
probe.sparsity_analysis()
# probe.plot_entropy()
# probe.check_variance()