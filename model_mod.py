import torch
import torch.nn.functional as F
from transformers import XGLMTokenizer, XGLMForCausalLM
import re

class model_load():    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = XGLMForCausalLM.from_pretrained(model_name,cache_dir="transformers_cache")
        self.tokenizer = XGLMTokenizer.from_pretrained(model_name,cache_dir="transformers_cache")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
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
        # Find the last character in the sentence   
        pattern = r'[^\w\s]'
        # Find the last character in the sentence
        last_character = re.findall(pattern, sentence[-1])

        # pattern = r'[^a-zA-Z0-9\s]'
        pattern = r'[^\w\sáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]'
        cleaned_sentence = re.sub(pattern, '', sentence)

        if last_character:
            sent_temp = cleaned_sentence
            sentence = "<s> " + sent_temp + last_character[0] + " </s>"
        else:
            sentence = "<s> " + cleaned_sentence + ". </s>"
        return sentence

    def autoregressive_formatting(self, sentence):
        # # one_shot_prompt = "<s> You are a language model that can predict the next word given a context. </s> <s> "
        one_shot_prompt = ""
        sentence = one_shot_prompt + self.modify_sentence(sentence)
        prefixes = []
        predictions = []
        for i in range(1,len(sentence.split())):
            prefix = ' '.join(sentence.split()[:i]).strip()
            prefixes.append(prefix)
            pred = sentence.split()[i]
            predictions.append(pred)
        return prefixes, predictions
    
    def process_activations_detectors(self, ffn_hooks):
        activations = []
        expectations = []
        for key in ffn_hooks.keys():
            key_name = str(key).split(".")[-1]
            if key_name == "fc1":
                fc1 = ffn_hooks[key]
                trg = fc1[0,-1,:].unsqueeze(0).detach().cpu()
                activation = torch.relu(fc1[0,-1,:]).unsqueeze(0).detach().cpu()
                activations.append(activation)
                det_act = torch.softmax(fc1[0,-1,:],dim=-1).unsqueeze(0).detach().cpu()
                expectation = trg * det_act
                expectations.append(expectation)
        #         activations.append(expectation)
        stacked_tensor = activations[0]
        for i in range(1,len(activations)):
            if len(stacked_tensor.shape)==1:
                stacked_tensor = stacked_tensor.unsqueeze(0)
            if len(activations[i].shape)==1:
                activations[i] = activations[i].unsqueeze(0)
            stacked_tensor = torch.cat((stacked_tensor, activations[i]), dim=0)
        stacked_tensor2 = expectations[0]
        for i in range(1,len(expectations)):
            if len(stacked_tensor2.shape)==1:
                stacked_tensor2 = stacked_tensor2.unsqueeze(0)
            if len(expectations[i].shape)==1:
                expectations[i] = expectations[i].unsqueeze(0)
            stacked_tensor2 = torch.cat((stacked_tensor2, expectations[i]), dim=0)
        return stacked_tensor, stacked_tensor2

    # def process_activations_combinators(self, ffn_hooks):
    #     activations = []
    #     layer_predictions = []
    #     for key in ffn_hooks.keys():
    #         key_name = str(key).split(".")[-1]
    #         if key_name == "fc2":
    #             fc2 = ffn_hooks[key]
    #             # det_comb = torch.softmax(fc2[0,-1,:],dim=-1).detach().cpu()
    #             det_comb = torch.relu(fc2[0,-1,:]).detach().cpu()
    #             activations.append(det_comb)
    #             pred_comb = self.lm_layer(fc2[0,-1,:].unsqueeze(0))
    #             pred_comb = torch.argmax(pred_comb,dim=-1).detach().cpu()[0]
    #             pred_comb = self.tokenizer.decode([pred_comb])
    #             layer_predictions.append(pred_comb)
    #     stacked_tensor = activations[0]
    #     for i in range(1,len(activations)):
    #         if len(stacked_tensor.shape)==1:
    #             stacked_tensor = stacked_tensor.unsqueeze(0)
    #         if len(activations[i].shape)==1:
    #             activations[i] = activations[i].unsqueeze(0)
    #         stacked_tensor = torch.cat((stacked_tensor, activations[i]), dim=0)
    #     return stacked_tensor,layer_predictions

    def process_sentence(self, sentence):
        n_lyrs = self.model.config.num_layers
        ff_dim = self.model.config.ffn_dim
        prefixes, predictions  = self.autoregressive_formatting(sentence)
        detector_activations   = torch.zeros((n_lyrs,ff_dim))
        detector_expectations  = torch.zeros((n_lyrs,ff_dim))
        combinator_activations = torch.zeros((n_lyrs,ff_dim))
        for prefix, prediction in zip(prefixes, predictions):
            pred, ffn_hooks = self.model_forward(prefix.strip())
            act_detector, exp_detector   = self.process_activations_detectors(ffn_hooks)
        #     act_combinator, layer_preds = self.process_activations_combinators(ffn_hooks)
            detector_activations  += act_detector
            detector_expectations += exp_detector
        #     combinator_activations += act_combinator
        #    #print(f"{prefix}\t<--->\t{prediction}\t{layer_preds}\t{pred}")
        # # print("\n\n")
        return detector_activations, detector_expectations
        # return detector_activations, combinator_activations
    
    def model_forward(self, prefix):
        # Encode the sentence
        encoded_inputs = self.tokenizer(prefix, return_tensors="pt")
        encoded_inputs = encoded_inputs.to(self.device)
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token prediction
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = self.tokenizer.decode([next_token_id.item()])
        return next_token, self.activations