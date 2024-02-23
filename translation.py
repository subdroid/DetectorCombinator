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
from transformers import AutoModelForCausalLM, AutoTokenizer

from sacrebleu import sentence_bleu, corpus_bleu
from sacremoses import MosesTokenizer
import nltk
from evaluate import load

# from comet import Comet

import json
import copy
import sys
import torch 

# model_list = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]
# model_list = ["facebook/xglm-4.5B","facebook/xglm-2.9B","facebook/xglm-1.7B","facebook/xglm-564M"]
model_list = ["facebook/xglm-7.5B"]
from transformers import BitsAndBytesConfig

class TranslationLM():
  def __init__(self, model_name, device):
    self.model_name = model_name
    self.device = device
    # self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir="transformers_cache")
    # self.tokenizer = XGLMTokenizer.from_pretrained(model_name,cache_dir="transformers_cache")
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.model.to(self.device)
    # self.model.eval()
    # double_quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True)
    # self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=double_quant_config,
    #              low_cpu_mem_usage=True, cache_dir="transformers_cache")
    # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
    #             torch_dtype=torch.float16,low_cpu_mem_usage=True, cache_dir="transformers_cache")
    # # self.model.to(self.device)
    # self.model.eval()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
    self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      quantization_config=bnb_config,
                                                      cache_dir="transformers_cache",
                                                      device_map="cuda:0")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="transformers_cache")

  def lobotomize(self,lyr=None, model_loc=None, freeze_percent=None, langpair=None, category=None):
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
    freeze_loc = os.path.join(freeze_loc,str(freeze_percent))
    multi_freeze = os.path.join(freeze_loc,fl)
    
    model = self.freeze_data_load(multi_freeze,lyr)
    return model
  
  def freeze_data_load(self,f_name,lyr):
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
    return model

  def translate(self, model,input_text,result):
    
    with torch.no_grad():
      input_ids = self.tokenizer(input_text, padding=True, return_tensors="pt").to(self.device)["input_ids"]
      inputs = self.tokenizer(input_text, padding=True, return_tensors="pt").to(self.device)
      result_id = self.tokenizer(result, padding=True, return_tensors="pt").to(self.device)["input_ids"]
      res_len = input_ids.shape[1] + result_id.shape[1] + 10
      print(res_len)
      # output_ids = model.generate(input_ids, max_length=res_len, num_beams=5, temperature=1)
      # output_ids = model.generate(input_ids=input_ids,max_length=res_len,temperature=1) 
      output_ids = model.generate(**inputs, max_length=res_len)

      # output_ids = model.generate(input_ids, do_sample=True, max_length=res_len, num_beams=5,
      #                             repetition_penalty=0.6, top_k=50, top_p=0.95, temperature=0.5)
      print(output_ids) 

      # output_ids = model.generate(input_ids, max_length=res_len, num_beams=5)
      result = output_ids[0][input_ids.shape[1]:]
      output_text = self.tokenizer.decode(result, skip_special_tokens=True)
      # print(output_text)
      # print("works")
    return output_text
  
  def czeng_data(self):
    with open("data/cz_en_test", "r") as f:
        czeng_data = f.readlines()
    cs = [line.strip().split("<\t>")[0] for line in czeng_data]
    en = [line.strip().split("<\t>")[1] for line in czeng_data]
    return cs,en
# def calculate_bleu(translations, references):
#   bleu_score = corpus_bleu(translations, [references])
  # return bleu_score.scor
def make_folder(loc):
  if not os.path.exists(loc):
    os.mkdir(loc)
def run_translation(save_path  = None, model_obj=None, lobotomized_model=None, lang1=None, lang2=None, data_l1=None, data_l2=None, freeze=None, freeze_category=None):
  if not lobotomized_model:
    model_lm = copy.deepcopy(model_obj.model)
    # model_lm = model_obj.model
  else:
    model_lm = lobotomized_model
    
  src_l1, src_l2 = [], []
  mte_l2, mte_l1 = [], []
  rfe_l2, rfe_l1 = [], []
  
  loc_l1 = os.path.join(save_path,f"{lang1}2{lang2}")

  file_l1 = open(loc_l1,"w")
  print("source\treference\ttranslation\tBLEU\tG-BLEU\tCOMET",file=file_l1)
  file_l1.close()
  loc_l2 = os.path.join(save_path,f"{lang2}2{lang1}")
  file_l2 = open(loc_l2,"w")
  print("source\treference\ttranslation\tBLEU\tG-BLEU\tCOMET",file=file_l2)
  file_l2.close()

  bleu = load("bleu")
  comet_metric = load('comet')
  gbleu = load('google_bleu')
  
  count = 0

  for i in tqdm(range(len(data_l1))):
    actual_l1 = data_l1[i].strip()
    actual_l2 = data_l2[i].strip()
    # print(actual_l1)
    # print(actual_l2)

    example_prompt_l1 = ""
    example_prompt_l2 = ""

    for e in range(i+1,i+3):
      example_prompt_l1 += f" {lang1}: {data_l1[e]}\n{lang2}: {data_l2[e]}\n"
      example_prompt_l2 += f" {lang2}: {data_l2[e]}\n{lang1}: {data_l1[e]}\n"
    
    # print(example_prompt_l1)
    # print(example_prompt_l2)

    prompt2l2 = example_prompt_l1+ f"{lang1}: {actual_l1}\n{lang2}: "
    prompt2l1 = example_prompt_l2+ f"{lang2}: {actual_l2}\n{lang1}: "
    
    # print(prompt2l2)
    # print(prompt2l1)

    mt_l2 = model_obj.translate(model_lm, prompt2l2,actual_l2)
    print(mt_l2)
    # try:  
    #   print("elo")
    #   mt_l2 = model_obj.translate(model_lm, prompt2l2,actual_l2)
      
    #   bleu_score = bleu.compute(predictions=[mt_l2], references=[actual_l2])['bleu']
    #   comet_score = comet_metric.compute(predictions=[mt_l2], references=[actual_l2], sources=[actual_l1])['mean_score']
    #   gbleu_score = gbleu.compute(predictions=[mt_l2], references=[actual_l2])['google_bleu']

    #   src_l1.append(actual_l1)
    #   mte_l2.append(mt_l2)
    #   rfe_l2.append(actual_l2)
    #   file_l1 = open(loc_l1,"a")  
    #   print(f"{actual_l1}\t{actual_l2}\t{mt_l2}\t{bleu_score}\t{gbleu_score}\t{comet_score}",file=file_l1)
    #   # print(f"{actual_l1}\t{actual_l2}\t{mt_l2}\t{bleu_score}\t{gbleu_score}\t{comet_score}")
    #   file_l1.close()

    #   mt_l1 = model_obj.translate(model_lm, prompt2l1,actual_l1)
    #   bleu_score = bleu.compute(predictions=[mt_l1], references=[actual_l1])['bleu']
    #   comet_score = comet_metric.compute(predictions=[mt_l1], references=[actual_l2], sources=[actual_l1])['mean_score']
    #   gbleu_score = gbleu.compute(predictions=[mt_l1], references=[actual_l1])['google_bleu']
      
    #   src_l1.append(actual_l2)
    #   mte_l1.append(mt_l1)
    #   rfe_l1.append(actual_l1)
    #   file_l2 = open(loc_l2,"a")  
    #   print(f"{actual_l2}\t{actual_l1}\t{mt_l1}\t{bleu_score}\t{gbleu_score}\t{comet_score}",file=file_l2)
    #   # print(f"{actual_l2}\t{actual_l1}\t{mt_l1}\t{bleu_score}\t{gbleu_score}\t{comet_score}")
    #   file_l2.close()

    #   count += 1
    # except:
    #   print("Error in translation")
    #   continue  
    break
  #   if count==5:
  #     break
  
  # corp_gbleu_l2 = gbleu.compute(predictions=mte_l2, references=rfe_l2)['google_bleu']
  # corp_gbleu_l1 = gbleu.compute(predictions=mte_l1, references=rfe_l1)['google_bleu']
  # corp_bleu_l2  = bleu.compute(predictions=mte_l2, references=rfe_l2)['bleu']
  # corp_bleu_l1  = bleu.compute(predictions=mte_l1, references=rfe_l1)['bleu']
  del model_lm
  # return corp_bleu_l1, corp_bleu_l2, corp_gbleu_l1, corp_gbleu_l2
  # return None, None, None, None

if __name__ == "__main__":
  
  bleu_res = os.path.join(os.getcwd(),"bleu_scores_modified")
  if not os.path.exists(bleu_res):
    bleu_file = open(bleu_res,"w")
    print("model \t translation direction \t l_lob \t freeze_per \t category \t score(BLEU_Google) \t score (BLEU)",file=bleu_file)
    bleu_file.close()
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_file = model_list[int(sys.argv[1])]
  model_name = model_file.split("/")[-1]
  ffn_dic = {'xglm-564M':4096, 'xglm-1.7B':8192, 'xglm-2.9B':8192, 'xglm-4.5B':16384, 'xglm-7.5B':16384}
  topk = [0.0,0.1,0.2,0.5]
  langpair = ["cseng"]
  category = ["multilingual","l1","l2"]
  ffn_dim = ffn_dic[model_name]
  for f in topk:
    freeze = int(f*ffn_dim)
    for cat in category:
        translator = TranslationLM(model_file, device)
        num_layers = translator.model.config.num_hidden_layers
        cs,en = translator.czeng_data()
        for lp in langpair:
          if lp=="cseng":
            l1 = "czech"
            l2 = "english"
            fol_path = str(freeze)
            cat_path = cat
            translations = os.path.join(os.getcwd(),"translations_newexp")
            make_folder(translations)
            sv_path = os.path.join(translations,f"{model_name}")
            make_folder(sv_path)
            sv_path = os.path.join(sv_path,f"{fol_path}")
            make_folder(sv_path)
            sv_path = os.path.join(sv_path,f"{cat_path}")
            make_folder(sv_path)                                    
          if str(freeze)=="0":
            bleul1,bleul2,gbleul1,gbleul2=run_translation(save_path = sv_path, model_obj=translator, lang1="czech", lang2="english", data_l1=cs, data_l2=en, freeze=None, freeze_category=None)
          #   bleu_file = open(bleu_res,"a")
          #   print(f"{model_name} \t {l1}-{l2} \t 0 \t 0 \t {cat} \t {bleul2} \t {gbleul2}",file=bleu_file)
          #   print(f"{model_name} \t {l2}-{l1} \t 0 \t 0 \t {cat} \t {bleul1} \t {gbleul1}",file=bleu_file)
          #   bleu_file.close()
          #   torch.cuda.empty_cache()
          # else:            
          #   lobotomized = int(f*ffn_dim)
          #   for l in range(num_layers):
          #     model_lobotomized = translator.lobotomize(lyr=l, model_loc=model_name, 
          #                       freeze_percent=lobotomized, langpair=lp, category=cat)
          #     l_path = os.path.join(sv_path,f"{l}")
          #     make_folder(l_path)
          #     bleul1, bleul2, gbleul1, gbleul2 =run_translation(save_path = l_path, model_obj=translator,
          #                                   lobotomized_model= model_lobotomized, 
          #                                   lang1="czech", lang2="english", 
          #                                   data_l1=cs, data_l2=en, freeze=str(f), 
          #                                   freeze_category=cat)
          #     bleu_file = open(bleu_res,"a")
          #     print(f"{model_name} \t {l1}-{l2} \t {l} \t {f} \t {cat} \t {bleul2} \t {gbleul2}",file=bleu_file)
          #     print(f"{model_name} \t {l2}-{l1} \t {l} \t {f} \t {cat} \t {bleul1} \t {gbleul1}",file=bleu_file)
          #     bleu_file.close() 
          #     del model_lobotomized
          #     torch.cuda.empty_cache()