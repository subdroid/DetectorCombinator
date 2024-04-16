import torch
import torch.nn.functional as F
from transformers import XGLMTokenizer, XGLMForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys
import os
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import numpy as np
import csv
import concurrent.futures
import shutil
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')  

model_list = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]

class model_load():    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=".cache")

        nf4_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules= ['lm_head'],
                    )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=nf4_config,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=".cache"
                                                    )
        except ImportError:
            self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir=".cache")
                                            
        self.model.eval()

        self.activations = {}  # Dictionary to store layer activations

        # Define hook function to store activations
        def hook_fn(module, input, output, layer_name):
            self.activations[layer_name] = output
        
        for name, module in self.model.named_modules():
            if name.split(".")[-1]=="fc1" or name.split(".")[-1]=="fc2":
                module.register_forward_hook(
                    lambda module, input, output, layer_name=name: hook_fn(module, input, output, layer_name)
                )
        
        self.lm_weights = self.model.lm_head.weight.data.T
        self.lm_layer   = self.model.lm_head
       
    def modify_sentence(self,sentence):
        sentence = sentence.strip()
        # Find the last character in the sentence   
        pattern = r'[^\w\s]'
        last_character = re.findall(pattern, sentence[-1])
        pattern = r'[^\w\sáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]'
        cleaned_sentence = re.sub(pattern, '', sentence)
        if last_character:
            sent_temp = cleaned_sentence
            # sentence = "<s> " + sent_temp + " " + last_character[0] + " </s>"
            sentence = sent_temp + " " + last_character[0]
        else:
            # sentence = "<s> " + cleaned_sentence + ". </s>"
            entence = cleaned_sentence
        return sentence

    def autoregressive_formatting(self, sentence):
        sentence =  self.modify_sentence(sentence)
        prefixes = []
        predictions = []
        for i in range(1,len(sentence.split())):
            prefix = ' '.join(sentence.split()[:i]).strip()
            prefixes.append(prefix)
            pred = sentence.split()[i]
            predictions.append(pred)
        return prefixes, predictions
    
    def process_activations_detectors(self, ffn_hooks):
        representations = []
        for key in ffn_hooks.keys():
            key_name = str(key).split(".")[-1]
            if key_name == "fc1":
                fc1 = ffn_hooks[key]
                trg = fc1[0,-1,:].unsqueeze(0).detach().cpu()
                representations.append(trg)
        stacked_tensor = representations[0]
        for i in range(1,len(representations)):
            if len(stacked_tensor.shape)==1:
                stacked_tensor = stacked_tensor.unsqueeze(0)
            if len(representations[i].shape)==1:
                activations[i] = representations[i].unsqueeze(0)
            stacked_tensor = torch.cat((stacked_tensor, representations[i]), dim=0)
        return stacked_tensor
    
    def process_activations_combinators(self, ffn_hooks):
        representations = []
        for key in ffn_hooks.keys():
            key_name = str(key).split(".")[-1]
            if key_name == "fc2":
                fc2 = ffn_hooks[key]
                trg = fc2[0,-1,:].unsqueeze(0).detach().cpu()
                representations.append(trg)
        stacked_tensor = representations[0]
        for i in range(1,len(representations)):
            if len(stacked_tensor.shape)==1:
                stacked_tensor = stacked_tensor.unsqueeze(0)
            if len(representations[i].shape)==1:
                activations[i] = representations[i].unsqueeze(0)
            stacked_tensor = torch.cat((stacked_tensor, representations[i]), dim=0)
        return stacked_tensor

    
    def process_sentence(self, sentence):
        n_lyrs = self.model.config.num_layers
        ff_dim = self.model.config.ffn_dim
        model_dim = self.model.config.d_model
        
        prefixes, predictions    = self.autoregressive_formatting(sentence)
        detector_activations     = torch.zeros((n_lyrs,ff_dim))
        combinator_activations   = torch.zeros((n_lyrs,model_dim))
        det = []
        comb = []   
        for prefix, prediction in zip(prefixes, predictions):
            pred, ffn_hooks = self.model_forward(prefix.strip())
            act_detector   = self.process_activations_detectors(ffn_hooks)
            act_combinator = self.process_activations_combinators(ffn_hooks)
            det.append(act_detector)
            comb.append(act_combinator)
        return det, comb
            
    def model_forward(self, prefix):
        encoded_inputs = self.tokenizer(prefix, return_tensors="pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded_inputs = encoded_inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token prediction (greedy)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = self.tokenizer.decode([next_token_id.item()])
        return next_token, self.activations


def process_language(lang,sent_list,fol_path):
    count = 0
    for sent in tqdm(sent_list):
        detector, combinator = model_obj.process_sentence(sent)
        update_activations(count,detector,combinator,fol_path,lang)
        # count+=1
        # if count==1000:
        #     break
        # break

def update_activations(count,act,com,file_path,lang):
    det = os.path.join(file_path,"detectors")
    if not os.path.exists(det):
        os.mkdir(det)
    det = os.path.join(det,lang)
    if not os.path.exists(det):
        os.mkdir(det)
    
    for pid,prefix in enumerate(act):
        for lyr in range(prefix.shape[0]):
            lyr_loc = os.path.join(det,f"{lyr}")
            if not os.path.exists(lyr_loc):
                os.mkdir(lyr_loc)
            file_name = f"{count}_{pid}.npy"
            file_loc = os.path.join(lyr_loc,file_name)
            np.save(file_loc,prefix[lyr])

    comb = os.path.join(file_path,"combinators")
    if not os.path.exists(comb):
        os.mkdir(comb)
    comb = os.path.join(comb,lang)
    if not os.path.exists(comb):
        os.mkdir(comb)
    
    for pid,prefix in enumerate(com):
        for lyr in range(prefix.shape[0]):
            lyr_loc = os.path.join(comb,f"{lyr}")
            if not os.path.exists(lyr_loc):
                os.mkdir(lyr_loc)
            file_name = f"{count}_{pid}.npy"
            file_loc = os.path.join(lyr_loc,file_name)
            np.save(file_loc,prefix[lyr])

model_obj = model_load(model_list[int(sys.argv[1])])
model_name = model_list[int(sys.argv[1])].split("/")[-1]
num_layers = model_obj.model.config.num_layers
num_hidden = model_obj.model.config.ffn_dim
num_model  = model_obj.model.config.d_model

f_path = os.path.join(os.getcwd(),"model_snapshots") 
if not os.path.exists(f_path):
    os.mkdir(f_path)
fol_path = os.path.join(f_path,model_name)
if not os.path.exists(fol_path):
    os.mkdir(fol_path)

data_loc = os.path.join(os.getcwd(), 'data')
ec_cs_file = open(os.path.join(data_loc, 'csen_cs.txt'),"r").readlines()
eh_hi_file = open(os.path.join(data_loc, 'hien_hi.txt'),"r").readlines()
ed_de_file = open(os.path.join(data_loc, 'deen_de.txt'),"r").readlines()
ef_fr_file = open(os.path.join(data_loc, 'fren_fr.txt'),"r").readlines()
process_language("cs",ec_cs_file,fol_path)
process_language("hi",eh_hi_file,fol_path)
process_language("de",ed_de_file,fol_path)
process_language("fr",ef_fr_file,fol_path)
ec_en_file = open(os.path.join(data_loc, 'csen_en.txt'),"r").readlines()
eh_en_file = open(os.path.join(data_loc, 'hien_en.txt'),"r").readlines()
ed_en_file = open(os.path.join(data_loc, 'deen_en.txt'),"r").readlines()
ef_en_file = open(os.path.join(data_loc, 'fren_en.txt'),"r").readlines()
process_language("Encs",ec_en_file,fol_path)
process_language("Enhi",eh_en_file,fol_path)
process_language("Ende",ed_en_file,fol_path)
process_language("Enfr",ef_en_file,fol_path)