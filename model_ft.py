import torch
import torch.nn.functional as F
from transformers import XGLMTokenizer, XGLMForCausalLM, XGLMModel
from transformers import DataCollatorForLanguageModeling, AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import re
import sys
import os
from tqdm import tqdm
import transformers
from transformers import BitsAndBytesConfig, TrainerCallback
import numpy as np
import csv
import concurrent.futures
import shutil
import random
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')  

class CustomTrainer(Trainer):
    def __init__(self, *args, t_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_layer = t_layer
    """
    Reusing the loss computation to include multilingual contrastive loss
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        hidden_states = outputs['hidden_states']
        target_embeddings = hidden_states[self.t_layer]
        batch_size = target_embeddings.size(0)
        embeddings_norm = F.normalize(target_embeddings, p=2, dim=-1)  # Normalize embeddings 
        embeddings_flat = embeddings_norm.view(embeddings_norm.size(0), -1)

        # Apply K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings_flat.cpu().numpy())

        # Convert cluster labels to PyTorch tensor
        cluster_labels_tensor = torch.tensor(cluster_labels)
        c1, c2 = [], []
        for lid, lbl in enumerate(cluster_labels_tensor):
            if lbl==0:
                c1.append(embeddings_norm[lid])
            else:
                c2.append(embeddings_norm[lid])
        C1 = torch.stack(c1)
        C2 = torch.stack(c2)

        c1_centroid = torch.mean(C1,dim=0)
        c2_centroid = torch.mean(C2,dim=0)
        
        dists = torch.cdist(c1_centroid, c2_centroid)
        loss_contrastive = 1/torch.mean(dists)
        # print(loss_contrastive)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"]+loss_contrastive if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss

# Define your custom dataset class
class EvalDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.text = open(file_path, "r", encoding="utf-8").read().split("\n")[:1000]
        self.examples = tokenizer(self.text, return_tensors="pt", truncation=True, padding=True)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples["input_ids"][idx],
            "attention_mask": self.examples["attention_mask"][idx]
        }
    
class TrainLossCallback(TrainerCallback):
    def __init__(self, log_frequency=10, tlayer=None, tmode=None):
        self.log_frequency = log_frequency
        self.t_layer = tlayer
        self.t_mode  = tmode

        sloc = os.path.join(os.getcwd(),"train_csv")
        if not os.path.exists(sloc):
            os.makedirs(sloc)
        self.save_loc =  os.path.join(sloc,self.t_mode)
        if not os.path.exists(self.save_loc):
            os.makedirs(self.save_loc)


        ploc = os.path.join(os.getcwd(),"train_plots")
        if not os.path.exists(ploc):
            os.makedirs(ploc)
        self.pic_loc = os.path.join(ploc,self.t_mode)
        if not os.path.exists(self.pic_loc):
            os.makedirs(self.pic_loc)

        fl_loss = pd.DataFrame(columns=["step","loss"])
        self.f_name = f"train_loss_{self.t_layer}.csv"  
        self.f_loc = os.path.join(self.save_loc,self.f_name) 
        fl_loss.to_csv(self.f_loc,index=False)

        fl_loss = pd.DataFrame(columns=["loss"])
        self.f_name2 = "train_loss.csv"  
        self.f_loc2 = os.path.join(self.save_loc,self.f_name2)
        if not os.path.exists(self.f_loc2):
            fl_loss.to_csv(self.f_loc2,index=False)
        # fl_loss.to_csv(self.f_loc2,index=False)
        
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if (step % self.log_frequency==0) and (step!=self.log_frequency):
            loss = state.log_history[-1]['loss']

            fl = pd.read_csv(self.f_loc)
            df_temp = pd.DataFrame.from_dict({"step":[int(step)],"loss":[loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.f_loc,index=False)
            # fl.to_csv("train_loss.csv",index=False)
            plt.plot(fl["step"].tolist(),fl["loss"].tolist())
            plt.title("2.9B: Train Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            f_name = f"train_loss_{self.t_layer}.png"     
            f_loc  = os.path.join(self.pic_loc,f_name) 
            plt.savefig(f_loc)
            plt.close()
            plt.clf()
            
            fl = pd.read_csv(self.f_loc2)
            df_temp = pd.DataFrame.from_dict({"loss":[loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.f_loc2,index=False)
            # fl.to_csv("train_loss.csv",index=False)
            X = np.arange(0,len(fl["loss"].tolist()))
            plt.plot(X,fl["loss"].tolist())
            plt.title("2.9B: Train Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            f_name = f"train_loss_total_history.png"     
            f_loc  = os.path.join(self.pic_loc,f_name) 
            plt.savefig(f_loc)
            plt.close()
            plt.clf()

class EvalLossCallback(TrainerCallback):
    def __init__(self, tokenizer=None, model=None, eval_frequency=1, tlayer=None, tmode = None):
        self.eval_frequency = eval_frequency
        self.min_loss = 1000000
        self.eval_loc = os.path.join(os.getcwd(),"data")
        self.hn_eval  = os.path.join(self.eval_loc,"hien_hi.txt")
        self.en_eval  = os.path.join(self.eval_loc,"hien_en.txt")
        self.hi_dataset    = EvalDataset(self.hn_eval, tokenizer) 
        self.en_dataset    = EvalDataset(self.en_eval, tokenizer)
        self.model = model
        self.min_loss = 10000000
        self.tlayer = tlayer
        self.tmode  = tmode

        fl_loss = pd.DataFrame(columns=["step","loss"])
        
        eval_plt = "eval_plots"
        if not os.path.exists(eval_plt):
            os.makedirs(eval_plt)
        self.eval_plt = os.path.join(eval_plt,self.tmode)
        if not os.path.exists(self.eval_plt):
            os.makedirs(self.eval_plt)
            
        sv_eval_loc = "eval_csv"
        if not os.path.exists(sv_eval_loc):
            os.makedirs(sv_eval_loc)
        self.sv_eval_loc = os.path.join("eval_csv",self.tmode)
        if not os.path.exists(self.sv_eval_loc):
            os.makedirs(self.sv_eval_loc)
        self.ev_name = f"eval_loss_{self.tlayer}.csv"  
        self.ev_loc = os.path.join(self.sv_eval_loc,self.ev_name) 
        fl_loss.to_csv(self.ev_loc,index=False)

        fl_loss = pd.DataFrame(columns=["loss"])
        self.ev_loc2 = os.path.join(self.sv_eval_loc,f"eval_loss_history.csv") 
        if not os.path.exists(self.ev_loc2):
            fl_loss.to_csv(self.ev_loc2,index=False)

        # fl_loss.to_csv("eval_loss.csv",index=False)
        fl_loss = pd.DataFrame(columns=["step","loss"])
        
        sv_en_loc = "en_csv"
        if not os.path.exists(sv_en_loc):
            os.makedirs(sv_en_loc)
        self.sv_en_loc = os.path.join("en_csv",self.tmode)
        if not os.path.exists(self.sv_en_loc):
            os.makedirs(self.sv_en_loc)
        self.en_name = f"eval_loss_{self.tlayer}.csv"  
        self.en_loc = os.path.join(self.sv_en_loc,self.en_name)         
        fl_loss.to_csv(self.en_loc,index=False)
        # fl_loss.to_csv("en_loss.csv",index=False)

        sv_hi_loc = "hi_csv"
        if not os.path.exists(sv_hi_loc):
            os.makedirs(sv_hi_loc)
        self.sv_hi_loc = os.path.join("hi_csv",self.tmode)
        if not os.path.exists(self.sv_hi_loc):
            os.makedirs(self.sv_hi_loc)
        self.hi_name = f"eval_loss_{self.tlayer}.csv"  
        self.hi_loc = os.path.join(self.sv_hi_loc,self.hi_name)  
        fl_loss.to_csv(self.hi_loc,index=False)       
        # fl_loss.to_csv("hi_loss.csv",index=False)
        
        mod_loc = os.path.join(os.getcwd(),"saved_models")
        if not os.path.exists(mod_loc):
            os.makedirs(mod_loc)
        self.mod_loc = os.path.join(mod_loc,self.tmode)
        if not os.path.exists(self.mod_loc):
            os.makedirs(self.mod_loc)

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step==0 or (state.global_step%self.eval_frequency==0):
            self.model.eval()
            total_loss = 0.0
            en_dataloader = DataLoader(self.en_dataset, batch_size=8, shuffle=False)
            hi_dataloader = DataLoader(self.hi_dataset, batch_size=8, shuffle=False)
            en_loss = 0.0
            hi_loss = 0.0
            with torch.no_grad():
                for batch in en_dataloader:
                    input_ids = batch["input_ids"].to(self.model.device)
                    attention_mask = batch["attention_mask"].to(self.model.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    total_loss += loss.item()
                    en_loss += loss.item()
                for batch in hi_dataloader:
                    input_ids = batch["input_ids"].to(self.model.device)
                    attention_mask = batch["attention_mask"].to(self.model.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    hi_loss += loss.item()
                    total_loss += loss.item()
            total_loss = total_loss
            if total_loss<self.min_loss:
                self.min_loss = total_loss
                model_name = f"best_model"
                model_loc = os.path.join(self.mod_loc,model_name)
                if os.path.exists(model_loc):
                    shutil.rmtree(model_loc)
                self.model.save_pretrained(model_loc)
            en_loss = en_loss 
            hi_loss = hi_loss 

            # fl = pd.read_csv("eval_loss.csv")
            fl = pd.read_csv(self.ev_loc)
            df_temp = pd.DataFrame.from_dict({"step":[int(step)],"loss":[total_loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.ev_loc,index=False)
            # fl.to_csv("eval_loss.csv",index=False)

            plt.plot(fl["step"].tolist(),fl["loss"].tolist())
            plt.title("2.9B: Eval Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            f_name = f"eval_loss_{self.tlayer}.png"     
            f_loc = os.path.join(self.eval_plt,f_name)
            plt.savefig(f_loc)
            # plt.savefig("eval_loss.png")
            plt.close()
            plt.clf()
            
            fl = pd.read_csv(self.ev_loc2)
            df_temp = pd.DataFrame.from_dict({"loss":[total_loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.ev_loc2,index=False)
            # fl.to_csv("eval_loss.csv",index=False)
            X = np.arange(0,len(fl["loss"].tolist()))
            plt.plot(X,fl["loss"].tolist())
            plt.title("2.9B: Eval Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            f_name = f"eval_loss_total_history.png"     
            f_loc = os.path.join(self.eval_plt,f_name)
            plt.savefig(f_loc)
            # plt.savefig("eval_loss.png")
            plt.close()
            plt.clf()


            fl = pd.read_csv(self.en_loc)
            df_temp = pd.DataFrame.from_dict({"step":[int(step)],"loss":[en_loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.en_loc,index=False)
            # fl.to_csv("en_loss.csv",index=False)
            plt.plot(fl["step"].tolist(),fl["loss"].tolist(),label="English")
            fl = pd.read_csv(self.hi_loc)
            df_temp = pd.DataFrame.from_dict({"step":[int(step)],"loss":[hi_loss]})
            if not fl.empty:
                fl = pd.concat([fl,df_temp],ignore_index=True)
            else:
                fl = df_temp
            fl.to_csv(self.hi_loc,index=False)
            # fl.to_csv("hi_loss.csv",index=False)

            plt.plot(fl["step"].tolist(),fl["loss"].tolist(),label="Hindi")
            plt.legend()
            plt.title("2.9B: Eval Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            f_name = f"lang_eval_loss_{self.tlayer}.png"     
            f_loc = os.path.join(self.eval_plt,f_name)
            plt.savefig(f_loc)
            # plt.savefig("lang_eval_loss.png")
            plt.close()
            plt.clf()
            
            self.model.train()        

class PMDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        f1 = os.path.join(data_dir, "hindi")
        f2 = os.path.join(data_dir, "english")
        self.files1 = [os.path.join(f1, file) for file in os.listdir(f1)]
        self.files2 = [os.path.join(f2, file) for file in os.listdir(f2)]
        self.files = self.files1 + self.files2
        random.shuffle(self.files)
        self.current_file_idx = 0
        self.current_chunk_idx = 0
        self.current_chunk = None
        self.chunk_size = random.randint(50, 1000)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        self.load_next_chunk()
        return self.current_chunk
        # return self.tokenize_and_pad(self.current_chunk)

    def load_next_chunk(self):
        with open(self.files[self.current_file_idx], 'r', encoding='utf-8') as f:
            text = f.read()
        start = self.current_chunk_idx * self.chunk_size
        end = min((self.current_chunk_idx + 1) * self.chunk_size, len(text))
        self.current_chunk = text[start:end]
        if end == len(text):
            self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
            self.current_chunk_idx = 0
        else:
            self.current_chunk_idx += 1

class model_load():    
    def __init__(self, model_name, mode, t_layer):
        print("ok")
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name,cache_dir=".cache")
        self.dataset    = PMDataset("/home/bhattacharya/personal_work_troja/Detector_exp", self.tokenizer)

        # print("ok")
        self.mode    = mode
        self.t_layer = t_layer

        self.saved_m_loc = os.path.join(os.getcwd(),"saved_models")
        if not os.path.exists(self.saved_m_loc):
            os.makedirs(self.saved_m_loc)
        self.saved_model_loc = os.path.join(self.saved_m_loc,f"{mode}")
        if not os.path.exists(self.saved_model_loc):
            os.makedirs(self.saved_model_loc)
        
        model_exists = len(os.listdir(self.saved_model_loc))>0
        try:
            if not model_exists:
                self.model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir=".cache", output_hidden_states=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(os.path.join(self.saved_model_loc,"best_model"), output_hidden_states=True)
        except ImportError:
            if not model_exists:
                self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir=".cache", output_hidden_states=True)                                         
            else:
                self.model = XGLMForCausalLM.from_pretrained(os.path.join(self.saved_model_loc,"best_model"), output_hidden_states=True)  

        print(self.model)

        for name, param in self.model.named_parameters():
            print(name)
        # MoE_model = _moe_convert(16,self.model,self.tokenizer)

        # # Freeze all all layers
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # # Unfreeze selected layers (LIFT)
        # for name, param in self.model.named_parameters():
        #     if name.split(".")[0]=="model":
        #         if len(name.split("."))==5:
        #             lyr_name = name.split(".")[2]
        #             if int(lyr_name) == self.t_layer:
        #                 param.requires_grad = True
        #     elif name.split(".")[0]=="layers":
        #         if len(name.split("."))==4:
        #             lyr_name = name.split(".")[1]
        #             if int(lyr_name) == self.t_layer:
        #                 param.requires_grad = True
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()

        # # Define optimizer hyperparameters
        # self.optimizer_kwargs = {
        #     "lr": 7.5e-4,
        #     "eps": 1e-8,
        #     # "weight_decay": 0.01,
        #     "betas": (0.9, 0.98)
        # }

    def _get_training_arguments(self, **kwargs):
        default_training_args = {
            "output_dir": "ft_models",
            "logging_dir": "ft_logs",
            "logging_steps": 1,
            "overwrite_output_dir": True,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 8,
            "learning_rate": 6e-5,
            # "learning_rate": 5e-5,
            # "learning_rate": 5e-4,
            "gradient_accumulation_steps": 2,
            "auto_find_batch_size": True,
            "include_tokens_per_second": True,
            **kwargs
        }
        return TrainingArguments(**default_training_args)    

    def custom_collate(self,batch=None):
        tokens =self.tokenizer(batch, padding=True, return_tensors="pt")      
        return {'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"], 'labels': tokens["input_ids"]}

    def fine_tune(self):        
        
        train_dataset = self.dataset
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
            )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.optimizer_kwargs
        )
        
        total_steps = self._get_training_arguments().num_train_epochs*(len(train_dataset)/self._get_training_arguments().per_device_train_batch_size)
        total_steps = int(total_steps)

        scheduler=transformers.get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=10,num_training_steps=total_steps)


        optimizers = optimizer, scheduler
        

        # Define callbacks
        print_loss_callback = TrainLossCallback(log_frequency=10, tlayer=self.t_layer, tmode = self.mode)
        eval_loss_callback  = EvalLossCallback(tokenizer=self.tokenizer, model=self.model, eval_frequency=200, tlayer=self.t_layer, tmode = self.mode)

        # trainer = Trainer(
        #     model=self.model,
        #     args=self._get_training_arguments(),
        #     data_collator=self.custom_collate,
        #     train_dataset=train_dataset,
        #     optimizers=optimizers,
        #     callbacks=[print_loss_callback,eval_loss_callback],
        # )

        trainer = CustomTrainer(
            model=self.model,
            args=self._get_training_arguments(),
            data_collator=self.custom_collate,
            train_dataset=train_dataset,
            optimizers=optimizers,
            callbacks=[print_loss_callback,eval_loss_callback],
            t_layer = self.t_layer,
        )
        
    
        trainer.train()
        
        del self.model

class Expert(nn.Module):
    def __init__(self,ffn_dim=None,n_experts=None,d_model=None):
        super(Expert,self).__init__()
        self.L1 = nn.Linear(ffn_dim,n_experts)
        self.activation = nn.ReLU()
        self.L2 = nn.Linear(n_experts,d_model)
    def forward(input_rep):
        return self.L2(self.activation(self.L1(input_rep))) 


def _moe_convert(num_experts,mother_model,tokenizer):
    num_layers = mother_model.config.num_layers
    for name, params in mother_model.named_parameters():
        print(name)
    

# class MoEModel(nn.Module):
#     def __init__(self,num_experts,mother_model,tokenizer):
#         super(MoEModel,self).__init__()
#         self.num_experts = num_experts
#         self.base_model = mother_model
#         self.num_layers = self.base_model.config.num_layers
#         self.expert_layer = Expert(self.base_model.config.ffn_dim,num_experts,self.base_model.config.d_model)
#         self.expert_ff_layers = nn.ModuleList([self.expert_layer for _ in range(self.num_layers)])
        
#     def forward(self):
#         for name, params in self.base_model.named_parameters():
#             print(name)



    # def _get_expert_ffn_layers(self):
    #     for i in range(self.num_hidden_layers):
    #         layer = nn.Sequential(
    #             nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size),
    #             nn.ReLU(),
    #             nn.LayerNorm(self.base_model.config.hidden_size)
    #         )
    #         expert_ff_layers.append(layer)
    #     return expert_ff_layers


model = model_load("facebook/xglm-564M","ascending",20)

# for i in range(1,27):
#     beg = 19
#     end = beg+i
#     model = model_load("facebook/xglm-2.9B","ascending",end)
#     model.fine_tune()
#     torch.cuda.empty_cache()
#     # break

# for i in range(26,0,-1):
#     end = 19
#     beg = end+i
#     model = model_load("facebook/xglm-2.9B","descending",beg)
#     model.fine_tune()
#     torch.cuda.empty_cache()

# D=[]
# loop = True
# for i in range(26):
#     while loop:
#         r = random.randint(20,45)
#         if r not in D:
#             D.append(r)
#             loop = False
#     loop = True
#     model = model_load("facebook/xglm-2.9B","random",r)
#     model.fine_tune ()
#     torch.cuda.empty_cache()

# # print(sorted(D),len(D))
