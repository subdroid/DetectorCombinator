#!/usr/bin/env python
import torch
from model_mod import model_load
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import subprocess
import os
from tqdm import tqdm 
import csv
import shutil

from transformers import XGLMTokenizer, XGLMForCausalLM

from sacrebleu import corpus_bleu, sentence_bleu
from sacremoses import MosesTokenizer
# from comet import Comet

import json
import copy

# model_list = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]
model_list = ["facebook/xglm-1.7B"]

class TranslationLM():
  def __init__(self, model_name, device):
    self.model_name = model_name
    self.device = device
    self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir="transformers_cache2")
    self.tokenizer = XGLMTokenizer.from_pretrained(model_name,cache_dir="transformers_cache2")
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.model.eval()

  def lobotomize(self,lyr, model_loc, freeze_percent, langpair, category):
    if langpair == "cseng":
      if category == "multilingual":
        fl = "multilingual_csen.json"
      elif category == "l1":
        fl = "cs.json"
      elif category == "l2":
        fl = "en.json"
    elif langpair == "hieng":
      if category == "multilingual":
        fl = "multilingual_hien.json"
      elif category == "l1":
        fl = "hi.json"
      elif category == "l2":
        fl = "en.json"
    mod_loc = os.path.join("neuron_lists",model_loc)
    list_loc = os.path.join(os.getcwd(),mod_loc,"activation")
    freeze_loc = os.path.join(list_loc,str(langpair))
    if freeze_percent != 0:
      freeze_loc = os.path.join(freeze_loc,str(freeze_percent))
      multi_freeze = os.path.join(freeze_loc,fl)
    else: 
      multi_freeze = None
    model = self.freeze_data_load(multi_freeze,lyr)
    return model
  
  def freeze_data_load(self,f_name,lyr):
    if f_name is not None:
      # Opening a JSON file for writing
      with open(f_name, 'r') as file:
          data = json.load(file)
      model = copy.deepcopy(self.model)
      for name, param in model.named_parameters():
            name_split = name.split(".")
            if name_split[1]=="layers":
                if name_split[2]==str(lyr):
                  layer = str(int(name_split[2])+1)
                  keys_freeze = data[layer]['keys']
                  if name_split[3]=="fc1":
                      if name_split[4]=="weight":
                        for k in keys_freeze:
                          param.data[int(k),:] = 0.0
                      elif name_split[4]=="bias":
                        for k in keys_freeze:
                          param.data[int(k)] = 0.0
    else:
       model = copy.deepcopy(self.model)
    return model
  
  def translate(self, model,input_text,result):
    with torch.no_grad():
      input_ids = self.tokenizer(input_text, padding=True, return_tensors="pt").to(self.device)["input_ids"]
      result_id = self.tokenizer(result, padding=True, return_tensors="pt").to(self.device)["input_ids"]
      res_len = input_ids.shape[1] + result_id.shape[1] + 10
      # res_len = result_id.shape[1] + 10
      # output_ids = self.model.generate(input_ids, max_length=res_len, num_beams=5, length_penalty=0.6)
      output_ids = model.generate(input_ids, do_sample=True, max_length=res_len, num_beams=5) 
    #                               # repetition_penalty=0.6, 
    #                               # temperature=0.5, top_k=50, top_p=0.95,)
    #   # , temperature=0.8)
    #   # , length_penalty=0.6)
      result = output_ids[0][input_ids.shape[1]:]
      output_text = self.tokenizer.decode(result, skip_special_tokens=True)
      print(output_text)
    # return output_text

  def czeng_data(self):
    with open("data/cz_en_test", "r") as f:
        czeng_data = f.readlines()
    cs = [line.strip().split("<\t>")[0] for line in czeng_data]
    en = [line.strip().split("<\t>")[1] for line in czeng_data]
    return cs,en

def calculate_bleu(translations, references):
  bleu_score = corpus_bleu(translations, [references])
  # return bleu_score.score

def run_translation(translator,prompt_lang1,prompt_lang2,l1,l2,f,lp,cat):
  # # src_l1, src_l2 = [], []
  # # mte_l2, mte_l1 = [], []
  # # rfe_l2, rfe_l1 = [], []
  for i in tqdm(range(len(l1))):
    actual_l1 = l1[i]
    actual_l2 = l2[i]     
    example_prompt_l1 = ""
    example_prompt_l2 = ""
    for e in range(i+1,i+3):
        example_prompt_l1 += f"{l1[e]} : {l2[e]}\n"
        example_prompt_l2 += f"{l2[e]} : {l1[e]}\n"
        print(example_prompt_l1)
        print(actual_l1)
  #       # model.translate(model_lm, example_prompt_l1,actual_l1)  
  # #       break
  # #   # self.translate(model_lm, example_prompt_l1,actual_l1)
  # #   # mt_l1, mt_l2 = translation_direction(model_lm, [example_prompt_l1,example_prompt_l2],[actual_l1,actual_l2])
  # #   break  
  
if __name__ == "__main__":
  freeze = [0]
  langpair = ["cseng"]
  category = ["multilingual"]
  bleu_res = os.path.join(os.getcwd(),"bleu_scores")
  bleu_file = open(bleu_res,"w")
  # print("model \t translation direction \t l_lob \t freeze_per \t category \t score",file=bleu_file)
  for model in tqdm(model_list):
    # if os.path.exists("transformers_cache2"):
    #     shutil.rmtree("transformers_cache2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model.split("/")[-1]
    translator = TranslationLM(model, device)
    num_layers = translator.model.config.num_hidden_layers
    cs,en = translator.czeng_data()
    for f in freeze:
      print(f)
      # for lp in langpair:
      #   for cat in category:
      #     if f==0:
            # model_lm = copy.deepcopy(translator.model)
            # run_translation(translator, "czech", "english", cs, en, f, lp, cat)
          