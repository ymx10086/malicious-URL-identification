import pandas as pd

df = pd.read_csv('./data/test.csv')

import sys,os
import torch
import argparse

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.utils.data import Dataset,DataLoader,RandomSampler, SequentialSampler, Sampler
from typing import List,Tuple

from tqdm import trange,tqdm

def collate_input_features(batch):
    # print(len(batch))
    max_seq_len = max(len(x[0]) for x in batch)
    sz = len(batch)

    # print(max_seq_len)
    input_ids = np.zeros((sz, max_seq_len), np.int64)
    labels = np.zeros((sz),np.int64)
    for i,ex in enumerate(batch):
        # print(ex)
        assert(len(ex[0])<=max_seq_len)
        input_ids[i, :len(ex[0])] = ex[0]
        labels[i]= ex[1] 
    input_ids = torch.as_tensor(input_ids)
    labels = torch.as_tensor(labels)

    return input_ids,labels


class BertClassifer(nn.Module):
    def __init__(self, bert_path, n_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dense = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768,2048),
            nn.GELU(),
            nn.Linear(2048, n_labels)
        )
        self.n_labels = n_labels
    def forward(self, seqs, features,):
        _, pooled = self.bert(seqs, output_all_encoded_layers=False)
        logits = self.dense(pooled)
        return logits
    
class InputFeatureDataset(Dataset):
    def __init__(self, examples, labels):
        self.exmaples = examples
        self.labels = labels
    def __getitem__(self, index):
        return (self.exmaples[index],self.labels[index])
    def __len__(self):
        return len(self.exmaples)

def build_train_dataloader(data,labels,config):
    ds = InputFeatureDataset(data,labels)
    return DataLoader(ds, batch_size=config['batch_size'], sampler=RandomSampler(ds),collate_fn=collate_input_features)

label_dict = {
"正常网址":0,
"购物消费":1, 
"婚恋交友":2, 
"假置身份":3,
"钓鱼网站":4,
"冒充公检法":5,
"平台诈骗":6,
"招聘兼职":7,
"杀猪盘":8,
"博彩赌博":9,
"信贷理财":10,
"刷单诈骗":11,
"中奖":12
}

def main():

    output_dir = "./model/"

    config = {
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64 ,
        'num_train_epochs': 3,
    }
    df = pd.read_csv('./data/train2.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    features = []
    labels = []
    for (id,url,text,label) in df.itertuples():
        url = " ".join(url.split("."))
        text = str(text)
        text += url 
        if not str.isdigit(label):
            label = label_dict[label]
        labels.append(label)
        tokens = tokenizer.tokenize(text)
        if len(tokens) > 512:
            tokens = tokens[:512]
        ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        features.append(ids[0])

    model = BertModel.from_pretrained('bert-base-chinese',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    model = model.to(config['device'])
    loss_fn = CrossEntropyLoss()
    loss_fn = loss_fn.to(config['device'])
    optimizer = BertAdam(model.parameters(),lr=5e-5,
                        max_grad_norm=1.0)

    # ans = collate_input_features([([1,2,3],1),([1,2,3,4],2)])
    # print(ans)

    train_dataloader = build_train_dataloader(features, labels, config)
    
    model.train()
    loss_ema = 0
    total_steps = 0
    decay = 0.99
    loss = []

    for _ in trange(config['num_train_epochs'], desc = 'Epoch', ncols=100):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0,0
        pbar = tqdm(train_dataloader, desc='loss', ncols = 100)

        for step, Batch in enumerate(pbar):
            inputs_ids, labels = Batch
            inputs_ids = inputs_ids.to(config['device'])
            labels = labels.to(config['device'])
            logits = model(inputs_ids)
            loss = loss_fn(logits,labels)

            total_steps+=1
            loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1-decay)
            descript = "loss=%.4f" % (loss_ema / (1-decay ** total_steps))
            pbar.set_description(descript,refresh=False)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())


if __name__ == "__main__":
    main()