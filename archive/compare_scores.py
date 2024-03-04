import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class plot_scores:
    def __init__(self):
        self.translation_folder = os.path.join(os.getcwd(), 'translations_newexp')
        # self.path = path
        # self.df = pd.read_csv(self.path)
    def navigate(self):
        return None
    def plot(self):
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
        
        for folder in os.listdir(self.translation_folder):
            fol_loc = os.path.join(self.translation_folder, folder)
            print(folder)

        return None

plotter = plot_scores()
plotter.plot()