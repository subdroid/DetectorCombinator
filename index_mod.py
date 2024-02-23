import os
import csv
import pandas as pd
import numpy as np 
import json
from matplotlib import pyplot as plt    
from tqdm import tqdm
import numpy as np
import shutil
import math

def sort_index(matrix):
    sorted_matrix = {}
    for rid, row in enumerate(matrix):
        sorted_indices = [index for index in sorted(range(len(row)), key=lambda x: row[x], reverse=True)]
        sorted_values = [row[i] for i in sorted_indices]
        dictionary = dict(zip(sorted_indices, sorted_values))
        sorted_matrix[rid+1] = dictionary
    return sorted_matrix

def get_sorted_indices(fol_path,model,cat_type):
    # folder where the indices will be saved
    fol_ind = os.getcwd()
    folders = ["sorted_indices",model,cat_type]
    for item in folders:
        fol_ind = os.path.join(fol_ind,item)
        if not os.path.exists(fol_ind):
            os.mkdir(fol_ind)

    for lang in tqdm(os.listdir(fol_path)):
        print(f"processing {model}-->{lang}")
        # Leads to the csv file for a particular language
        lang_path = os.path.join(fol_path,lang)
        try:
            lang_cont = pd.read_csv(lang_path,header=None)
            lang_cont = lang_cont.to_numpy()
            sorted_dict = sort_index(lang_cont)
            file_name = lang.replace('.csv','.json')
            save_loc = os.path.join(fol_ind,file_name)
            # Save the dictionary as a JSON file
            with open(save_loc, 'w') as json_file:
                json.dump(sorted_dict, json_file, indent=2)
        except:
            continue    

def extract_set(num_neurons,l1,l2,name,langn1,langn2,loc):
    L1 = {}
    L2 = {}
    multiling = {}    
    
    Lang1, Lang2, Multi = [], [], []
    for k1,k2 in zip(l1.keys(),l2.keys()):
        lyr_l1 = l1[k1]
        lyr_l2 = l2[k2]
        l1_key = list(lyr_l1.keys())[:num_neurons]
        l2_key = list(lyr_l2.keys())[:num_neurons]
        intersection = list(set(l1_key).intersection(set(l2_key)))
        lang1 = list(np.setdiff1d(np.array(l1_key), np.array(intersection)))
        lang2 = list(np.setdiff1d(np.array(l2_key), np.array(intersection)))
        L1[k1] = {"count":len(lang1),"keys":lang1}
        L2[k2] = {"count":len(lang2),"keys":lang2}
        multiling[k1] = {"count":len(intersection),"keys":intersection}
        Lang1.append(len(lang1))
        Lang2.append(len(lang2))
        Multi.append(len(intersection))
    f_list = [name,num_neurons]
    for f in f_list:
        loc = os.path.join(loc,str(f))
        make_folder(loc)
    
    multiling_loc = os.path.join(loc,f"multilingual_{langn1}{langn2}.json")
    lang1_loc = os.path.join(loc,f"{langn1}.json")
    lang2_loc = os.path.join(loc,f"{langn2}.json")
    with open(multiling_loc, 'w') as json_file:
            json.dump(multiling, json_file, indent=2)
    with open(lang1_loc, 'w') as json_file:
            json.dump(L1, json_file, indent=2)
    with open(lang2_loc, 'w') as json_file:
            json.dump(L2, json_file, indent=2)
    
    return Lang1, Lang2, Multi

def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_json_file(input_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
    return data

def make_lists(model_name):
    """
    Source for the calculations:
    https://github.com/facebookresearch/fairseq/blob/main/examples/xglm/README.md
        model     layers 	model_dim ffn_dim 	Languages
    XGLM 564M 	  24 	1024 	   4096 	  30 
    XGLM 1.7B 	  24 	2048 	   8192 	  30 
    XGLM 2.9B 	  48 	2048 	   8192 	  30 
    XGLM 4.5B 	  48 	2048 	   16384 	  134
    XGLM 7.5B 	  32 	4096 	   16384 	  30
    """
    print(model_name.strip())
    divs = [0.01,0.05,0.1,0.15,0.2,0.5]
    mffn = [4096,8192,8192,16384,16384]
    if model_name.strip()=="xglm-564M":
        model_dim = 4096
        n_list = [int(model_dim*x) for x in divs]
    elif model_name.strip()=="xglm-1.7B":
        model_dim = 8192
        n_list = [int(model_dim*x) for x in divs]
    elif model_name.strip()=="xglm-2.9B":
        model_dim = 8192
        n_list = [int(model_dim*x) for x in divs]
    elif model_name.strip()=="xglm-4.5B":   
        model_dim = 16384
        n_list = [int(model_dim*x) for x in divs]
    elif model_name.strip()=="xglm-7.5B":
        model_dim = 16384
        n_list = [int(model_dim*x) for x in divs]
    return n_list

def extract_neuron_list():
    fol_ind = os.path.join(os.getcwd(),"neuron_lists")
    if not os.path.exists(fol_ind):
        os.mkdir(fol_ind)
    sort_ind = os.path.join(os.getcwd(),"sorted_indices")
    for model in tqdm(os.listdir(sort_ind)):
        model_loc = os.path.join(sort_ind,model)
        fol_ind_model = os.path.join(fol_ind,model)
        make_folder(fol_ind_model)
        for eval_cat in os.listdir(model_loc):
            eval_cat_loc = os.path.join(model_loc,eval_cat)
            fol_ind_eval = os.path.join(fol_ind_model,eval_cat)
            make_folder(fol_ind_eval)
            en_c = read_json_file(os.path.join(eval_cat_loc,"Encs.json"))
            cs   = read_json_file(os.path.join(eval_cat_loc,"cs.json"))
            n_list = make_lists(model)
            for e_max in n_list:
                Lang1,Lang2,Multi = extract_set(e_max,cs,en_c,"cseng","cs","en",fol_ind_eval)
                m = ""
                for i in range(len(Multi)):
                    m+=f"\t{Multi[i]}"
                # We assume that the maximum number of layers in the model is 50
                diff = 50-len(Multi)
                for i in range(diff):
                    m+=f"\t0"
                f_loc = os.path.join(os.getcwd(),"multi_ling_detectors.csv")
                fl = open(f_loc,"a")
                print(f"{model}\t{e_max}\t{eval_cat}{m}",file=fl)
                fl.close()
            try:
                en_h = read_json_file(os.path.join(eval_cat_loc,"Enhi.json"))
                hi   = read_json_file(os.path.join(eval_cat_loc,"hi.json"))
                for e_max in n_list:
                    Lang1,Lang2,Multi = extract_set(e_max,hi,en_h,"hieng","hi","en",fol_ind_eval)
            except FileNotFoundError:
                continue
 
def extract_numeric_part(s):
    # Use regular expression to extract numeric part
    import re
    match = re.match(r'xglm-(\d+(\.\d+)?)([BM])', s)
    if match:
        value = float(match.group(1))
        multiplier = match.group(3)
        if multiplier == 'B':
            return value * 1e9  # Convert to billion
        elif multiplier == 'M':
            return value * 1e6  # Convert to million

def plotter(pair,data):
    categories = ["activation"]
    cats = ["multilingual"]
    for cat in categories:
        for lang in cats:
            num_models = len(data.keys())
            mods = list(data.keys())
            sorted_model_names = sorted(mods, key=extract_numeric_part)
            for i, model in enumerate(sorted_model_names):
                model_dic = data[model][cat]
                for lang_pair in model_dic.keys(): 
                    if lang_pair=="cseng":
                        pair_name = "Czech-English"
                    elif lang_pair=="hieng":
                        pair_name = "Hindi-English"
                    try:
                        pdata = model_dic[lang_pair]
                        fig, axes = plt.subplots(nrows=2, figsize=(12, 8), gridspec_kw={'hspace': 0.5})
                        fig.suptitle(f"{model}: Distribution of multilingual detectors ({pair_name})", fontsize=12)
                        # plt.title(f'{model}: Distribution of multilingual detectors ({pair_name})', fontsize=12)
                        for lid, l in enumerate(pdata.keys()):
                            if l==lang:
                                lang_obj = pdata[l]
                                keys = np.array([int(x) for x in lang_obj.keys()])
                                sorted_keys = np.sort(keys)
                                colors = ['blue','orange','purple','red','olive','cyan']
                                perctg = [0.01,0.05,0.1,0.15,0.2,0.5]
                                for kid,k in enumerate(list(sorted_keys)):
                                    color = colors[kid]
                                    k_ = k
                                    k = str(k)
                                    k_top = lang_obj[k]
                                    vals  = []
                                    vals2 = []
                                    for layers in k_top.keys():
                                        col = k_top[layers]
                                        count = col['count']
                                        keys  = col['keys']
                                        ratio = count/k_
                                        vals.append(count)
                                        vals2.append(ratio)
                                    indices = np.arange(len(vals))
                                    percentage = int(perctg[kid]*100)
                                    axes[0].scatter(indices, vals, marker='o', s=5, c=color, label=f'Top {percentage} %')
                                    axes[0].plot(indices, vals, linestyle='-', linewidth=1, color=color, alpha=0.5)  
                                    axes[0].legend()
                                    axes[0].grid()
                                    axes[0].set_xlabel('Layers')
                                    axes[0].set_ylabel('Number of detectors')
                                    axes[0].set_title('Absolute number of multilingual detectors across layers')
                                    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                    
                                    axes[1].scatter(indices, vals2, marker='o', s=6, c=color, label=f'Top {percentage} %')
                                    axes[1].plot(indices, vals2, linestyle='--', linewidth=1.5, color=color, alpha=0.5)  
                                    axes[1].legend()
                                    axes[1].grid()
                                    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')                                     
                                    axes[1].set_xlabel('Layers')
                                    axes[1].set_ylabel('Ratio of multilingual detectors')
                                    # # Fit a smoothed curve using numpy.polyfit
                                    # degree = 2  # Adjust the degree of the polynomial fit
                                    # coeffs = np.polyfit(indices, vals2, degree)
                                    # smoothed_vals = np.polyval(coeffs, indices)
                                    # # Plot the smoothed curve
                                    # axes[1].plot(indices, smoothed_vals, linestyle='solid', linewidth=1.5, color=color)
                                    axes[1].set_title(f'Ratio (w.r.t. top k) of multilingual detectors across layers.')
                        fol_name = os.path.join(os.getcwd(),"multiling_detectors")
                        if not os.path.exists(fol_name):
                            os.mkdir(fol_name)
                        plt.savefig(os.path.join(fol_name,f"{model}_{lang_pair}.png"), bbox_inches='tight')
                        plt.close()
                        plt.clf()
                    except KeyError:
                        continue

def plot_neuron_stats():
    fol_ind = os.path.join(os.getcwd(),"neuron_lists")
    data = {}
    for model in os.listdir(fol_ind):
        model_loc = os.path.join(fol_ind,model)
        data[model] = {}
        for eval_cat in os.listdir(model_loc):
            eval_loc = os.path.join(model_loc,eval_cat)
            data[model][eval_cat] = {}
            for lang_pair in os.listdir(eval_loc):
                if lang_pair=="hieng":
                    lang1 = "hi"
                    lang2 = "en"
                elif lang_pair=="cseng":
                    lang1 = "cs"
                    lang2 = "en"
                pair_loc = os.path.join(eval_loc,lang_pair)
                data[model][eval_cat][lang_pair] = {}
                for topk in os.listdir(pair_loc):
                    topk_loc = os.path.join(pair_loc,topk)
                    for pt in os.listdir(topk_loc):
                        cont = json.load(open(os.path.join(topk_loc,pt)))
                        if pt.split('_')[0]=="multilingual":
                            lang = "multilingual" 
                        else:
                            lang = pt.split('.')[0]
                            if lang=="en":
                                lang = "english"
                            elif lang=="cs":
                                lang = "czech" 
                            elif lang=="hi":
                                lang = "hindi"
                        if lang not in data[model][eval_cat][lang_pair].keys():
                            data[model][eval_cat][lang_pair][lang] = {}
                        data[model][eval_cat][lang_pair][lang][topk] = cont
    langpairs = ["cseng","hieng"]
    for p in langpairs:
        plotter(p,data)                                       

def bleu_plotter():
    bleu_file = os.path.join(os.getcwd(),"bleu_scores")
    bleu_fl = pd.read_csv(bleu_file,sep="\t")
    cats = bleu_fl[' category '].unique()
    freeze_per = bleu_fl[' freeze_per '].unique()
    translation_dir = bleu_fl[' translation direction '].unique()
    models = bleu_fl['model '].unique()
    fol_loc = os.path.join(os.getcwd(),"bleu_plots")
    if not os.path.exists(fol_loc):
        os.mkdir(fol_loc)
    for m in models:
        for f, per in enumerate(freeze_per):
            fig, axes = plt.subplots(nrows=len(translation_dir)*2, figsize=(12, 8), gridspec_kw={'hspace': 1.0,'wspace': 0.8})
            fig.suptitle(f"MT performance after lobotomizing top-{per} neurons in {m} model.", fontsize=16)
            for t, trans in enumerate(translation_dir):
                colors1 = ['lightcoral','green','royalblue']
                colors2 = ["red","green","blue"]
                for c, cat in enumerate(cats):
                    data =  bleu_fl[(bleu_fl[' category ']==cat) & (bleu_fl[' freeze_per ']==per) & (bleu_fl[' translation direction ']==trans) & (bleu_fl['model ']==m)]
                    bleu = data[' score'].to_numpy()            
                    if len(bleu)>1:
                        if cat.strip()=="multilingual":
                            ls = "dotted"
                        else:
                            ls = "dashed"
                        axes[2*t].plot(bleu,label=f"{cat}",linestyle=ls,linewidth=3,color=colors1[c])
                        axes[(2*t)].set_title(f"{trans}")
                        axes[(2*t)].grid(True)
                        axes[(2*t)].set_xlabel('Layers')
                        axes[(2*t)].set_ylabel('BLEU')
                        axes[(2*t)].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust the location of the legend
                        # # Fit a smoothed curve using numpy.polyfit
                        degree = 2  # Adjust the degree of the polynomial fit
                        coeffs = np.polyfit(np.arange(len(bleu)), bleu, degree)
                        smoothed_vals = np.polyval(coeffs, np.arange(len(bleu)))
                        ls = 'solid'
                        axes[(2*t)+1].plot(np.arange(len(bleu)), smoothed_vals, label=f"{cat}", linestyle=ls, linewidth=3, color=colors2[c])
                        axes[(2*t)+1].set_title(f"BLEU (linear fit): {trans}; coefficient={coeffs}")
                        axes[(2*t)+1].grid(True)
                        axes[(2*t)+1].set_xlabel('Layers')
                        axes[(2*t)+1].set_ylabel('BLEU')
                        axes[(2*t)+1].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust the location of the legend
            plt.savefig(os.path.join(fol_loc,f"{m}_{per}.png"), bbox_inches='tight')  # Use bbox_inches to include the legend in the saved figure
            plt.close()

def compare_bleu():
    f_loc = os.path.join(os.getcwd(),"bleu_scores")
    f_cnt = pd.read_csv(f_loc,sep="\t")
    topk = f_cnt[' freeze_per '].unique()
    models = f_cnt['model '].unique()
    directions = f_cnt[' translation direction '].unique()
    M={}
    for m in models:
        M[m] = int(extract_numeric_part(m))
    sorted_models = sorted(M, key=M.get)
    for did, d in enumerate(f_cnt[' category '].unique()):
        data = f_cnt[f_cnt[' category ']==d]
        for tid, t in enumerate(directions):
            tdata = data[data[' translation direction ']==t]
            for kid,k in enumerate(topk):
                # nrows = math.ceil(len(sorted_models)/2)
                if not k==0:
                    fig, axes = plt.subplots(nrows=1, figsize=(12, 8), gridspec_kw={'hspace': 1.0,'wspace': 0.8})
                    fig.suptitle(f"{t} BLEU after lobotomizing top-{k} {d} detectors", fontsize=16)    
                    kdata = tdata[tdata[' freeze_per ']==k]
                    baseline = tdata[tdata[' freeze_per ']==0]
                    colors = ['lightcoral','olive','firebrick','darkgreen','darkblue']
                    for mid, m in enumerate(sorted_models):
                        mdata = kdata[kdata['model ']==m]
                        bdata = baseline[baseline['model ']==m]
                        scores = mdata[' score'].to_numpy() 
                        axes.plot(scores, label=m,color=colors[mid])
                        bscore = bdata[' score'].to_numpy()
                        if bscore.shape[0]>0:
                            bscores = [] 
                            for i in range(len(scores)):
                                bscores.append(bscore[0])
                            axes.plot(bscores, label=f"{m}_baseline",ls="dashed",color=colors[mid])
                        axes.grid(True)
                        names = [str(x) for x in range(len(scores))]
                        axes.set_xticks(range(len(scores)), names)
                        axes.set_xlabel('Layers')
                        axes.set_ylabel('Ratio')
                        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust the location of the legend
                    fol_loc = os.path.join(os.getcwd(),f"bleu_plots_{t}_{d}_{k}")
                    plt.savefig(fol_loc, bbox_inches='tight')  # Use bbox_inches to include the legend in the saved figure
                    plt.close()    

def compare_multiling():
    f_loc = os.path.join(os.getcwd(),"multi_ling_detectors.csv")
    f_cnt = pd.read_csv(f_loc,sep="\t")
    ffn_dic = {'xglm-564M':4096, 'xglm-1.7B':8192, 'xglm-2.9B':8192, 'xglm-4.5B':16384, 'xglm-7.5B':16384}
    topk = [0.01,0.05,0.1,0.15,0.2,0.5]
    # topk = f_cnt['topk'].unique()
    # print(topk)
    models = f_cnt['model'].unique()
    M={}
    for m in models:
        M[m] = int(extract_numeric_part(m))
    sorted_models = sorted(M, key=M.get)
    for kid,k in enumerate(topk):
        nrows = math.ceil(len(models)/2)
        fig, axes = plt.subplots(nrows=nrows, figsize=(12, 8), gridspec_kw={'hspace': 1.0,'wspace': 0.8})
        fig.suptitle(f"Ratio of multilingual detectors for top-{int(k*100)}%", fontsize=16)
        # kdata = f_cnt[f_cnt['topk']==k]
        colors = ['lightcoral','olive','firebrick','darkgreen','darkblue']
        ls = ['dashed','dashed','dashed','solid','solid']
        lw = [2.5,3,3,4,4]
        for mid, m in enumerate(sorted_models):
            temp_data = f_cnt[f_cnt['model']==m]
            ffn_dim = ffn_dic[m]
            tk = int(ffn_dim*k)
            ddata = temp_data[temp_data['topk']==tk]
    #         ddata = kdata[kdata['model']==m]
            mdata = ddata[ddata['category']=="activation"].to_numpy()[0]
            m_name = mdata[0]
            m_tot  = int(mdata[1])
            plot_list = []
            for m_ in mdata[3:]:
                rat = int(m_)/m_tot
                if rat!=0:
                    plot_list.append(rat)
            axes[math.floor(mid/2)].plot(plot_list,linestyle=ls[mid], linewidth=lw[mid], label=m,color=colors[mid])
            axes[math.floor(mid/2)].grid(True)
            names = [str(x) for x in range(len(plot_list))]
            axes[math.floor(mid/2)].set_xticks(range(len(plot_list)), names)
            axes[math.floor(mid/2)].set_xlabel('Layers')
            axes[math.floor(mid/2)].set_ylabel('Ratio')
            axes[math.floor(mid/2)].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust the location of the legend
        fol_loc = os.path.join(os.getcwd(),"multiling_comparison_plots")
        if not os.path.exists(fol_loc):
            os.makedirs(fol_loc)
        fol_loc = os.path.join(fol_loc,f"multi_ratio_plots_{k}.png")
        # print(fol_loc)
        plt.savefig(fol_loc, bbox_inches='tight')  # Use bbox_inches to include the legend in the saved figure
        plt.close()    

if __name__ == '__main__':
    # folder_path = os.path.join(os.getcwd(),'results')
    # for model in os.listdir(folder_path):
    #     model_loc = os.path.join(folder_path, model)
    #     for folder in os.listdir(model_loc):
    #         cat = folder.split('_')[1]
    #         if cat=="exp":
    #             cat_type = "expectation"
    #         elif cat=="act":
    #             cat_type = "activation"
    #         folder_loc = os.path.join(model_loc, folder)
    #         get_sorted_indices(folder_loc,model,cat_type)

    # f_loc = os.path.join(os.getcwd(),"multi_ling_detectors.csv")
    # fl = open(f_loc,"w")
    # max_layers = 50
    # h=""
    # for l in range(max_layers):
    #     h+=f"\t{l+1}"
    # print(f"model\ttopk\tcategory{h}",file=fl)
    # fl.close()

    # extract_neuron_list()
    
    # plot_neuron_stats()
    # bleu_plotter()
    compare_multiling()
    # compare_bleu()
