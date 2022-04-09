import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from transformers.models.roberta.configuration_roberta import RobertaConfig
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch
import torch.nn as nn
import os

torch.distributed.init_process_group(backend="nccl")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# 统计词频
with open('dataset/keep_tokens/tokens_dict_3.pkl', 'rb') as l:
    tokens = pickle.load(l)
tokens[-1], tokens[-2], tokens[-3] = 2, 3, 4  # -1: cls, -2: sep, -3: mask


class RobertaDataset(Dataset):

    def __init__(self, corpus, vocab: dict, seq_len: int = 32):
        self.vocab = vocab
        self.seq_len = seq_len
        self.lines = corpus
        self.corpus_lines = self.lines.shape[0]

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2 = self.get_sentence(idx)
        text_pair = [-1] + text1 + [-2] + text2 + [-2]
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = [tokens.get(t, 1) for t in text_pair]
        segment_ids = [0] * len(token_ids)

        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        padding_label = [-100 for _ in range(self.seq_len + 3 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), output_ids.extend(padding_label), segment_ids.extend(padding)
        attention_mask = np.array(attention_mask)
        token_ids = np.array(token_ids)
        segment_ids = np.array(segment_ids)
        output_ids = np.array(output_ids)
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        t1, t2, _ = self.lines.iloc[idx].values
        t1 = [int(i) for i in t1.split(' ')]
        t2 = [int(i) for i in t2.split(' ')]
        self.truncate_sequences(self.seq_len, -1, t1, t2)
        if np.random.random() < 0.5:
            return t2, t1
        else:
            return t1, t2

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

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=7,
                                     rng=random.Random()):
        vocab_words = list(tokens.keys())
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == -1 or token == -2:
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = -3
                    output_labels[index] = tokens.get(output_tokens[index], 1)
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                        output_labels[index] = tokens.get(output_tokens[index], 1)
                    else:
                        masked_token = vocab_words[rng.randint(0, 21122)]  # 取不到特殊字符
                        output_labels[index] = tokens.get(output_tokens[index], 1)
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels


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


class MyRoberta(nn.Module):
    def __init__(self, config, keep_tokens=None, resume=False):
        super(MyRoberta, self).__init__()
        self.config = config
        if resume:
            self.roberta = BertModel(config=self.config)
        else:
            self.roberta = BertModel.from_pretrained('./roberta-cn-uer/pytorch_model.bin', config=self.config)

        self.config.vocab_size = 21128
        self.cls = BertOnlyMLMHead(self.config)

        if keep_tokens is not None:
            embedding = nn.Embedding(21128, 768)
            weight_roberta = torch.load('embeddingRobertaUer.pth')
            weight_roberta['weight'] = weight_roberta['weight'][keep_tokens]
            weight = nn.Parameter(weight_roberta['weight'])
            embedding.weight = weight
            self.roberta.embeddings.word_embeddings = embedding

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        masked_lm_labels = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            mask = (labels != -100)
            masked_lm_loss = loss_fct(prediction_scores[mask].view(-1, self.config.vocab_size), labels[mask].view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


def train(args, model):
    train_data = pd.read_csv(args.train_data_path, sep='\t', names=['text_a', 'text_b', 'label'])
    train_data_2 = pd.read_csv(args.round2_train_data_path, sep='\t', names=['text_a', 'text_b', 'label'])
    test_data_a = pd.read_csv(args.testA_data_path, sep='\t', names=['text_a', 'text_b', 'label'])
    test_data_b = pd.read_csv(args.testB_data_path, sep='\t', names=['text_a', 'text_b', 'label'])

    train_data['label'], test_data_a['label'], test_data_b['label'], train_data_2['label'] = -100, -100, -100, -100
    train_data = pd.concat([train_data, test_data_a, test_data_b, train_data_2], ignore_index=True)

    pretrain_dataset = RobertaDataset(train_data, tokens, args.max_seq_length)
    pretrain_sampler = DistributedSampler(pretrain_dataset)
    train_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, sampler=pretrain_sampler)
    total_steps = len(train_loader) * (args.epoch + 20)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = int(total_steps * args.warmup_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    fgm = None
    if args.attack_type == 'fgm':
        fgm = FGM(model)

    model.zero_grad()
    set_seed(args)

    for epoch in range(args.resume_epoch, args.epoch):
        pretrain_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader)
        losses = []
        for step, data in enumerate(pbar):
            optimizer.zero_grad()

            inputs = {
                'input_ids': data['input_ids'].cuda(args.local_rank, non_blocking=True).long(),
                'attention_mask': data['attention_mask'].cuda(args.local_rank, non_blocking=True).long(),
                'token_type_ids': data['token_type_ids'].cuda(args.local_rank, non_blocking=True).long(),
                'labels': data['output_ids'].cuda(args.local_rank, non_blocking=True).long()
            }

            masked_lm_loss = model(**inputs)[0]

            if args.gradient_accumulation_steps > 1:
                masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(masked_lm_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                masked_lm_loss.backward()

            if args.attack_type == 'fgm':
                fgm.attack(args.fgm_epsilon)
                loss_adv = model(**inputs)[0]
                loss_adv.backward()
                fgm.restore()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(masked_lm_loss.cpu().detach().numpy())
            pbar.set_description(
                f'epoch:{epoch + 1} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f}')

        if args.local_rank == 0:
            if (epoch + 1) == 150:
                torch.save(model.module.state_dict(), args.model_save_dir + f'uer-distribute-epoch{epoch + 1}.pth',
                           _use_new_zipfile_serialization=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', default='none', type=str, choices=['fgm', 'pgd', 'none'])
    parser.add_argument("--fgm_epsilon", default=0.2, type=float, help="fgm epsilon")
    parser.add_argument(
        "--train_data_path",
        default='./tcdata/gaiic_track3_round1_train_20210228.tsv',
        type=str,
        help="Path to data ",
    )
    parser.add_argument(
        "--round2_train_data_path",
        default='./tcdata/gaiic_track3_round2_train_20210407.tsv',
        type=str,
        help="Path to data ",
    )
    parser.add_argument(
        "--testA_data_path",
        default='./tcdata/gaiic_track3_round1_testA_20210228.tsv',
        type=str,
        help="Path to data ",
    )
    parser.add_argument(
        "--testB_data_path",
        default='./tcdata/gaiic_track3_round1_testB_20210317.tsv',
        type=str,
        help="Path to data ",
    )
    parser.add_argument(
        "--model_save_dir",
        default='./pretrain_model/',
        type=str,
        help="The output directory where the pretrained model checkpoints will be written.",
    )
    parser.add_argument(
        "--resume_model_path",
        default='',
        type=str,
        help="The path of the pretrained model checkpoints to resume.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--epoch", default=100, type=int, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--resume_epoch", default=0, type=int, help="the epoch where to resume",
    )
    parser.add_argument("--do_resume", action='store_true', default=False)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0.0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--bert_config_path",
        default='./roberta-cn-uer/config.json',
        type=str,
        help="Path to bert_config ",
    )
    parser.add_argument(
        "--keep_tokens_path",
        default='./dataset/keep_tokens/keep_tokens_3.txt',
        type=str,
        help="Path to keep_tokens ",
    )
    parser.add_argument("--my_vocab_size", type=int, default=21128)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")

    args = parser.parse_args()

    set_seed(args)

    torch.cuda.set_device(args.local_rank)


    config = RobertaConfig.from_json_file(args.bert_config_path)
    keep_tokens = []
    with open(args.keep_tokens_path, "r") as f:
        for token in f.readlines():
            keep_tokens.append(int(token))
    
    if args.do_resume:
        config.vocab_size = args.my_vocab_size
        model = MyRoberta(config=config, resume=True)
        model.load_state_dict(torch.load(args.resume_model_path))
        #print(model)
        model.cuda(args.local_rank)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train(args, model)
    else:
        model = MyRoberta(config=config, keep_tokens=keep_tokens)
        #print(model)
        model.cuda(args.local_rank)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train(args, model)


if __name__ == '__main__':
    main()
