import pandas as pd
import pickle
import argparse
import random
from time import time
from tqdm import tqdm
from model.modeling_nezha import NeZhaModel
from transformers import get_linear_schedule_with_warmup
from model.configuration_nezha import NeZhaConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from optimizer import Lookahead
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import os

train_data1 = pd.read_csv('./tcdata/gaiic_track3_round1_train_20210228.tsv', sep='\t',
                         names=['text_a', 'text_b', 'label'])
train_data2 = pd.read_csv('./tcdata/gaiic_track3_round2_train_20210407.tsv', sep='\t',
                         names=['text_a', 'text_b', 'label'])
train_data = pd.concat([train_data1, train_data2], ignore_index=True)


# 统计词频
with open('./dataset/keep_tokens/tokens_dict_3.pkl', 'rb') as l:
    vocab_dict = pickle.load(l)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
                       

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=32, num_block=2):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()
        # norms_1 = nn.ModuleList([LayerNorm(36) for _ in range(len(self.layers))])
        # norms_2 = nn.ModuleList([LayerNorm(9) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            # net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()


        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            # self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Model(nn.Module):
    def __init__(self, args, config):
        super(Model, self).__init__()
        self.args = args
        self.nezha = NeZhaModel(config=config)
        #self.fc_input_dims = 737  # cls
        self.fc_input_dims = 705  # cls
        if args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc_input_dims = 449
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc_input_dims = 449
        elif args.struc == 'idcnn':
            self.idcnn = IDCNN(input_size=768, filters=64)
            self.fc_input_dims = 33
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])
        self.fc = nn.Linear(self.fc_input_dims, args.num_classes)

    def forward(self, x):
        output = self.nezha(**x)[0]  # 0:sequence_output  1:pooler_output
        if self.args.struc == 'cls':
            output = output[:, 0, :]  # cls
            # output = output  # pooler
        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(output)
                last_hidden = hidden[0].permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(output)
                last_hidden = hidden.permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)
            elif self.args.struc == 'idcnn':
                output = self.idcnn(output)
                output = torch.mean(output, dim=1)
        if self.args.AveragePooling:
            if self.args.struc == 'idcnn':
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=32, stride=1).squeeze(1)
            else:
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.maxlen, stride=1).squeeze(1)
        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        output = torch.sigmoid(output)

        return output


class NeZhaDataset(Dataset):

    def __init__(self, corpus, vocab: dict, seq_len: int = 64, predict: bool = False):
        self.vocab = vocab
        self.seq_len = seq_len
        self.lines = corpus
        self.corpus_lines = self.lines.shape[0]
        self.predict = predict

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2, label = self.get_sentence_and_label(idx)
        text1_ids = [self.vocab.get(t, 1) for t in text1]
        text2_ids = [self.vocab.get(t, 1) for t in text2]

        token_ids = ([2] + text1_ids + [3] + text2_ids + [3])
        segment_ids = [0] * len(token_ids)

        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), segment_ids.extend(padding)
        attention_mask = np.array(attention_mask)
        token_ids = np.array(token_ids)
        segment_ids = np.array(segment_ids)
        label = np.array(label)
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "label": label}
        return output

    def get_sentence_and_label(self, idx):

        t1, t2, label = self.lines.iloc[idx].values
        if 0 == int(label):
            label = [1, 0]
        else:
            label = [0, 1]
        t1 = [int(i) for i in t1.split(' ')]
        t2 = [int(i) for i in t2.split(' ')]
        self.truncate_sequences(64, -1, t1, t2)

        if np.random.random() < 0.5 and not self.predict:
            return t2, t1, label
        else:
            return t1, t2, label

    def truncate_sequences(self, maxlen, index, *sequences):
        """截断总长度至不超过maxlen
        """
        sequences = [s for s in sequences if s]
        while True:
            lengths = [len(s) for s in sequences]
            if sum(lengths) > maxlen:
                i = np.argmax(lengths)
                sequences[i].pop(index)
            else:
                return sequences


def train(args, model, model_path):
    
    train_dataset = NeZhaDataset(train_data, vocab_dict, args.maxlen)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    best_auc = 0.
    # total_steps = len(train_loader) * epoch
    if args.lookahead:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                      eps=args.adam_epsilon)
        optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                      eps=args.adam_epsilon)

    criterion = nn.BCELoss()
    # warmup_steps = int(total_steps * 0.1)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=total_steps)
    fgm = FGM(model)
    model.zero_grad()

    for e in range(args.epoch):
        pbar = tqdm(train_loader)
        losses, acc_list = [], []
        for data in pbar:
            model.train()
            data['label'] = data['label'].to(args.device).float()
            optimizer.zero_grad()

            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            outputs = model(inputs)
            loss = criterion(outputs, data['label'])
            loss.backward()

            fgm.attack(epsilon=0.2)
            outputs_adv = model(inputs)
            loss_adv = criterion(outputs_adv, data['label'])
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            # scheduler.step()

            losses.append(loss.cpu().detach().numpy())
            output_array = outputs.cpu().detach().numpy()
            label_array = data['label'].cpu().detach().numpy()
            acc_list.extend(np.argmax(output_array, axis=1) == np.argmax(label_array, axis=1))
            pbar.set_description(
                f'epoch:{e + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.5f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

        torch.save(model.state_dict(), args.model_save_path + f'/wwm_{args.struc}_best_model.pth',
                       _use_new_zipfile_serialization=False)


def evaluate(model, data_loader, args):
    model.eval()
    true, positive_logits = [], []
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for data in pbar:
            data['label'] = data['label'].float()
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            outputs = model(inputs)
            positive_logit = outputs[:, 1] / (outputs.sum(axis=1) + 1e-8)
            true.extend(np.argmax(data['label'], axis=1))
            positive_logits.extend(positive_logit.cpu().numpy())
        auc_score = roc_auc_score(true, positive_logits)

    return auc_score


def predict(args, model):
    time_start = time()
    set_seed(args)

    test_data = pd.read_csv('./dataset/oppo_breeno_round1_data/gaiic_track3_round1_testB_20210317.tsv', sep='\t',
                            names=['text_a', 'text_b', 'label'])
    test_data['label'] = -100

    test_dataset = NeZhaDataset(test_data, vocab_dict, args.maxlen, predict=True)
    test_loader = DataLoader(test_dataset, 1)
    
    model.eval()
    with torch.no_grad():
        F = open(args.model_save_path + "/result.csv", 'w')
        for data in tqdm(test_loader):
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long()
            }
            outputs = model(inputs)
            positive_logit = outputs[:, 1] / (outputs.sum(axis=1) + 1e-8)
            for p in positive_logit:
                F.write('%f\n' % p)
        F.close()
    time_end = time()
    print(f'finish {time_end-time_start}s')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--maxlen", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epoch", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--struc', default='cls', type=str,
                        choices=['cls', 'bilstm', 'bigru', 'idcnn'])
    parser.add_argument("--model_save_path", default='./finetune_model', type=str,
                        help="Path to save finetuned model")
    parser.add_argument("--pre_model_path", default='./pretrain_model/', type=str)
    parser.add_argument("--bert_config", default='./nezha-cn-wwm/bert_config.json', type=str,
                        help="Path to save finetuned model")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lookahead", action='store_true',
                        help="Whether to use lookahead.")
    parser.add_argument("--dropout_num", default=1, type=int)
    parser.add_argument("--AveragePooling", default=True, type=bool)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lstm_dim", default=256, type=int)
    parser.add_argument("--gru_dim", default=256, type=int)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)

    args = parser.parse_args()
    set_seed(args)

    NeZhaconfig = NeZhaConfig.from_json_file(args.bert_config)
    NeZhaconfig.vocab_size = 21128
    model = Model(args=args, config=NeZhaconfig)

    if args.do_train:
        file_dir = args.pre_model_path
        file_list = os.listdir(file_dir)
        for name in file_list:
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and name.split('-')[0] == 'wwm':
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                #print(model)
                model = model.to(args.device)
                train(args, model, model_path)
    elif args.do_predict:
        model.load_state_dict(torch.load(args.model_save_path + '/best_model.pth', map_location='cuda'))
        model = model.to(args.device)
        predict(args, model)


if __name__ == '__main__':
    main()
