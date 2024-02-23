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

# model_list = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]
# model_list = ["facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]
model_list = ["facebook/xglm-7.5B"]
# model_list = ["facebook/xglm-2.9B", "facebook/xglm-4.5B"]

def data_select():
  hi_en = os.path.join(os.getcwd(),"data","hi_en_test")
  if not os.path.exists(hi_en):
    file_path = os.path.join(os.getcwd(),"data","parallel-n")
    get_random_lines_hien(file_path,20000)

  cs_en = os.path.join(os.getcwd(),"data","cz_en_test")
  if not os.path.exists(cs_en):
    file_path = os.path.join(os.getcwd(),"data","czeng20-test")
    get_random_lines_csen(file_path, 20000) 

def get_random_lines_csen(file_path, num_lines):
  csen_name = open(os.path.join(os.getcwd(),"data","cz_en_test"),"w")
  corp = []
  with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
              splits = line.split("\n")[0].split("\t")
              cs = splits[-2]
              en = splits[-1]
              if len(cs.split())>5:
                txt = f"{cs} <\t> {en}"
                corp.append(txt)        
  total_lines = len(corp)
  if total_lines <= num_lines:
    num_lines = total_lines
  # Generate a random sample of line numbers
  random_line_numbers = random.sample(range(1, total_lines), num_lines)
  c=0
  for l in random_line_numbers:
      ln = corp[l]
      print(ln,file=csen_name)    
      c+=1
  print(c)
  del corp, random_line_numbers

def get_random_lines_hien(folder_path,num_lines):
    hi_file = open(os.path.join(folder_path,"IITB.en-hi.hi")).read().split("\n")
    en_file = open(os.path.join(folder_path,"IITB.en-hi.en")).read().split("\n")
    hien_name = open(os.path.join(os.getcwd(),"data","hi_en_test"),"w")
    corp = []
    for hi,en in zip(hi_file,en_file):
      if hi.strip() and en.strip():
        if len(en.split())>5:
          item = f"{hi} <\t> {en}"
          corp.append(item)
    total_lines = len(corp)
    if total_lines <= num_lines:
      num_lines = total_lines
    # Generate a random sample of line numbers
    random_line_numbers = random.sample(range(1, total_lines), num_lines)
    c=0
    for l in random_line_numbers:
      ln = corp[l]
      print(ln,file=hien_name)
      c+=1
    print(c)
    del corp, random_line_numbers

def process_csen():
  cs_en = os.path.join(os.getcwd(),"data","cz_en_test")
  cs_en = open(cs_en).read().split("\n")
  Cs = []
  En = []
  for sent in cs_en:
    if sent.strip():
      if len(sent.split("<\t>"))==2:
        cs, en = sent.split("<\t>")
        Cs.append(cs)
        En.append(en)
  return Cs, En

def process_hien():
  hi_en = os.path.join(os.getcwd(),"data","hi_en_test")
  hi_en = open(hi_en).read().split("\n")
  Hi = []
  En = []
  for sent in hi_en:
    if sent.strip():
      if len(sent.split("<\t>"))==2:
        hi, en = sent.split("<\t>")
        Hi.append(hi)
        En.append(en)

  return Hi, En

def update_activations(act,file_path,lang):
  fol_path = os.path.join(file_path,"detector_act") 
  file_name = lang + ".csv"
  lang_file = os.path.join(fol_path,file_name)
  if not os.path.exists(lang_file):
    detector_activations = np.zeros((num_layers,num_hidden))
  else:
    detector_activations = np.loadtxt(lang_file, delimiter=',', ndmin=2)  
  detector_activations += act
  with open(lang_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(detector_activations)

def update_expectations(exp,file_path,lang):
  fol_path = os.path.join(file_path,"detector_exp") 
  file_name = lang + ".csv"
  lang_file = os.path.join(fol_path,file_name)
  if not os.path.exists(lang_file):
    detector_activations = np.zeros((num_layers,num_hidden))
  else:
    detector_activations = np.loadtxt(lang_file, delimiter=',', ndmin=2)  
  detector_activations += exp
  with open(lang_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(detector_activations)

def init_files(fol_path,lang):
  fol_path_act = os.path.join(fol_path,"detector_act") 
  if not os.path.exists(fol_path_act):
    os.mkdir(fol_path_act)
  fiile_name = lang + ".csv"
  lang_file = os.path.join(fol_path_act,fiile_name)
  if os.path.exists(lang_file):
    os.remove(lang_file)

  fol_path_exp = os.path.join(fol_path,"detector_exp") 
  if not os.path.exists(fol_path_exp):
    os.mkdir(fol_path_exp)
  fiile_name = lang + ".csv"
  lang_file = os.path.join(fol_path_exp,fiile_name)
  if os.path.exists(lang_file):
    os.remove(lang_file)
def process_language(lang,sent_list,fol_path):
  init_files(fol_path,lang)
  for sent in tqdm(sent_list):
    act_detector, exp_detector = model_obj.process_sentence(sent)
    update_activations(act_detector.numpy(),fol_path,lang)
    update_expectations(exp_detector.numpy(),fol_path,lang)
if __name__ == "__main__":
  for model in model_list:
    model_name = str(model).split("/")[-1]
    print(model_name)
    # model_obj  = model_load(model)
    # num_layers = model_obj.model.config.num_layers
    # num_hidden = model_obj.model.config.ffn_dim
    # data_select()
    # cs, en1 = process_csen()
    # hi, en2 = process_hien()

    # f_path = os.path.join(os.getcwd(),"results") 
    # if not os.path.exists(f_path):
    #   os.mkdir(f_path)
    # fol_path = os.path.join(f_path,model_name) 
    # if not os.path.exists(fol_path):
    #   os.mkdir(fol_path)
   
    # process_language("cs",cs,fol_path)
    # process_language("Encs",en1,fol_path)
    # process_language("hi",hi,fol_path)
    # process_language("Enhi",en2,fol_path)
    
    # shutil.rmtree('transformers_cache')
    # # break
