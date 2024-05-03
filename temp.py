import torch
import torch.nn.functional as F
from transformers import XGLMTokenizer, XGLMForCausalLM
from transformers import DataCollatorForLanguageModeling
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
import random
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')  


class PMDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        f1 = os.path.join(data_dir, "hindi")
        f2 = os.path.join(data_dir, "english")
        self.files1 = [os.path.join(f1, file) for file in os.listdir(f1)]
        self.files2 = [os.path.join(f2, file) for file in os.listdir(f2)]
        self.files = self.files1 + self.files2
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            text = f.read()
        return self.tokenizer(text.replace("\n"," "))


class model_load():    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=".cache")
        self.dataset = PMDataset("/home/bhattacharya/personal_work_troja/Detector_exp", self.tokenizer)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir=".cache")
        except ImportError:
            self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir=".c ache")                                         
        # Freeze all layers except layers 20-47
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if len(name.split("."))==5 or len(name.split("."))==6:
                lyr_name = name.split(".")[2]
                if int(lyr_name) in range(20,48):
                    param.requires_grad = True
        print("initiated")
    
    def fine_tune(self,batch_size=100,num_epochs=1):        
        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir="ft_model",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Define Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=lambda data: self.tokenizer(data, return_tensors="pt", padding=True, truncation=True),
            train_dataset=self.dataset,
        )
        # Train the model
        trainer.train()
        """
        # dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # # Define training arguments
        training_args = TrainingArguments(
            output_dir="finetuned",
            overwrite_output_dir=True,
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=4,  # Batch size per GPU/CPU
            logging_steps=1,
            save_steps=500,  # Save checkpoint every X steps
            save_total_limit=2,  # Limit the total amount of saved checkpoints
            learning_rate=5e-5,  # Learning rate
            weight_decay=0.01,  # Weight decay
            adam_epsilon=1e-8,  # Epsilon for Adam optimizer
            max_grad_norm=1.0,  # Max gradient norm
        )
        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        # Instantiate the Trainer class
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=None,
            tokenizer = self.tokenizer,
            train_function=self.my_training_function,
        )
        # Fine-tune the model
        trainer.train()
        """        
model = model_load("facebook/xglm-2.9B")
model.fine_tune(batch_size=20,num_epochs=1)
