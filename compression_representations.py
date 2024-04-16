import os
import subprocess
from tqdm import tqdm
import gzip
import numpy as np
import lzma
import shutil

def compress_file(input_file,output_file):
    # Load data from the .npy.gz file
    with gzip.open(input_file, 'rb') as f:
        data = np.load(f)    
    # # Save data to the .npy.xz file
    with lzma.open(output_file, 'wb') as f:
        np.save(f, data)

fl_path = os.path.join(os.getcwd(), 'snapshots')
for model in os.listdir(fl_path):
    model_loc = os.path.join(fl_path, model)
    for cat in os.listdir(model_loc):
        cat_loc = os.path.join(model_loc, cat)
        for lang in os.listdir(cat_loc):
            lang_loc = os.path.join(cat_loc, lang)
            for file in tqdm(os.listdir(lang_loc)):
                file_loc = os.path.join(lang_loc, file)
                if ".gz" in file:
                    output_file = file_loc.replace(".gz", ".xz")
                    compress_file(file_loc,output_file)
                    shutil.rmtree(file_loc)            