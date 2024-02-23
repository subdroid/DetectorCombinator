import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class compare_stats():
    def __init__(self) -> None:
        self.path_file = os.path.join(os.getcwd(), 'bleu_scores_modified')
        self.file_cont = pd.read_csv(self.path_file,sep='\t')
    def analyze(self):
        file_cols = {}
        for col in self.file_cont.columns:
            file_cols[col.strip()] = col
        categories = self.file_cont[file_cols['category']].unique()
        plot_loc = os.path.join(os.getcwd(), 'bleu_plots')
        if not os.path.exists(plot_loc):
            os.makedirs(plot_loc)    
        for cat in categories:
            cat_data = self.file_cont[self.file_cont[file_cols['category']] == cat]
            if cat.strip()=="l2":
                cat_ttl = "english"
            elif cat.strip()=="l1":
                cat_ttl = "czech"
            else:
                cat_ttl = "multilingual"
            ttl = f"BLEU (lobomized {cat_ttl}) " 
            directions = cat_data[file_cols['translation direction']].unique()
            for direction in directions:
                dir_data = cat_data[cat_data[file_cols['translation direction']] == direction]
                freeze_per = dir_data[file_cols['freeze_per']].unique()
                # cols = 1
                # rows = freeze_per.size // cols
                # fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
                fig, axs = plt.subplots(freeze_per.size-1, figsize=(9,8))
                fig.tight_layout(pad=5.0)  
                d_ttl = f"{ttl}: {direction}"
                count = -1
                for fid, freeze in enumerate(freeze_per):
                    freeze_data = dir_data[dir_data[file_cols['freeze_per']] == freeze]
                    models  = freeze_data[file_cols['model']].unique()
                    if freeze == 0.0:
                        baselines = {}
                        colors={}
                        cols = ['lightcoral','olivedrab','cyan','purple']
                        models = freeze_data[file_cols['model']].unique()
                        for mid, model in enumerate(models):
                            model_base_score = freeze_data[freeze_data[file_cols['model']]==model][file_cols['score (BLEU)']].tolist()[0]
                            baselines[model] = model_base_score
                            colors[model] = cols[mid]
                    else:
                        count+=1
                        # ax = axs[int(count/2),count%2]
                        ax = axs[count]
                        for mid, model in enumerate(models):
                            model_data = freeze_data[freeze_data[file_cols['model']] == model]
                            x_axis = [int(i)+1 for i in model_data[file_cols['l_lob']].tolist()]
                            y_axis = model_data[file_cols['score (BLEU)']].tolist()
                            bl = baselines[model]
                            baseline = [bl]*len(y_axis)
                            ax.plot(x_axis, y_axis, label=model, color=colors[model])
                            ax.plot(x_axis, baseline, color=colors[model], linestyle='--')
                            ax.set_title(f"Freeze percentage: {freeze*100} %")
                            ax.set_xlabel("Layers")
                            ax.set_ylabel("BLEU")
                            # ax.legend()
                            # ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3)  # Adjust vertical space here
                plt.suptitle(d_ttl, fontsize=16)
                plt.savefig(os.path.join(plot_loc, f"{cat_ttl}_{direction}.png"))


stats = compare_stats()
stats.analyze()