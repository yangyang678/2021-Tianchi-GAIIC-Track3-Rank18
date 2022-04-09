from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import pickle


def truncate_sequences(maxlen, index, *sequences):
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


def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in Tokenizer.stem(t):
                        if (
                            Tokenizer._is_cjk_character(c) or
                            Tokenizer._is_punctuation(c)
                        ):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            #truncate_sequences(32, -1, a, b)
            D.append((a, b, c))
    return D


def add_positive_instance():
    pos_pair = []
    with open('./oppo_breeno_round1_data/valid_data.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            if c == 1:
                pos_pair.append((a, b))

    print(len(pos_pair))
    pair_num = len(pos_pair)

    sen_with_index = []
    for pair in pos_pair:
        sen_with_index.append((pair[0], 0))  # 0表示句子对的第一句
    for pair in pos_pair:
        sen_with_index.append((pair[1], 1))  # 1表示句子对的第二句

    new = []  # 保存新样本
    lens = len(sen_with_index)
    for i in tqdm(range(lens)):
        for j in range(i + 1, lens):
            if j == i + pair_num:
                # 跳过同一句子对
                continue
            if sen_with_index[i][0] == sen_with_index[j][0]:
                if sen_with_index[i][1] == 0 and sen_with_index[j][1] == 0:
                    # 两句都是句子对中的第一句
                    if sen_with_index[i + pair_num][0] != sen_with_index[j + pair_num][0]:
                        new.append((sen_with_index[i + pair_num][0], sen_with_index[j + pair_num][0], 1))
                        # new.append((sen_with_index[j + pair_num][0], sen_with_index[i + pair_num][0], 1))
                elif sen_with_index[i][1] == 0 and sen_with_index[j][1] == 1:
                    # 第一句与第二句
                    if sen_with_index[i + pair_num][0] != sen_with_index[j - pair_num][0]:
                        new.append((sen_with_index[i + pair_num][0], sen_with_index[j - pair_num][0], 1))
                        # new.append((sen_with_index[j - pair_num][0], sen_with_index[i + pair_num][0], 1))
                elif sen_with_index[i][1] == 1 and sen_with_index[j][1] == 1:
                    # 都是第二句
                    if sen_with_index[i - pair_num][0] != sen_with_index[j - pair_num][0]:
                        new.append((sen_with_index[i - pair_num][0], sen_with_index[j - pair_num][0], 1))
                        # new.append((sen_with_index[j - pair_num][0], sen_with_index[i - pair_num][0], 1))

    print(len(new))
    new = set(new)
    print(len(new))

    with open('./oppo_breeno_round1_data/new_valid_positive_data.tsv', "w+") as f:
        for pair in new:
            f.write("{}\t{}\t{}\n".format(pair[0], pair[1], int(pair[2])))


def add_negative_instance():
    sen_pair = []
    with open('./oppo_breeno_round1_data/valid_data.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            sen_pair.append((a, b, c))

    print(len(sen_pair))
    pair_num = len(sen_pair)

    sen_with_index = []
    for pair in sen_pair:
        sen_with_index.append((pair[0], pair[2], 0))  # 0表示句子对的第一句
    for pair in sen_pair:
        sen_with_index.append((pair[1], pair[2], 1))  # 1表示句子对的第二句

    new = []  # 保存新样本
    lens = len(sen_with_index)
    for i in tqdm(range(lens)):
        for j in range(i + 1, lens):
            if j == i + pair_num:
                # 跳过同一句子对
                continue
            if sen_with_index[i][0] == sen_with_index[j][0]:
                if sen_with_index[i][1] == 1 and sen_with_index[j][1] == 0:
                    if sen_with_index[i][2] == 0 and sen_with_index[j][2] == 0:
                        # 两句都是句子对中的第一句
                        if sen_with_index[i + pair_num][0] != sen_with_index[j + pair_num][0]:
                            new.append((sen_with_index[i + pair_num][0], sen_with_index[j + pair_num][0], 0))
                            # new.append((sen_with_index[j + pair_num][0], sen_with_index[i + pair_num][0], 0))
                    elif sen_with_index[i][2] == 0 and sen_with_index[j][2] == 1:
                        # 第一句与第二句
                        if sen_with_index[i + pair_num][0] != sen_with_index[j - pair_num][0]:
                            new.append((sen_with_index[i + pair_num][0], sen_with_index[j - pair_num][0], 0))
                            # new.append((sen_with_index[j - pair_num][0], sen_with_index[i + pair_num][0], 0))
                    elif sen_with_index[i][2] == 1 and sen_with_index[j][2] == 1:
                        # 都是第二句
                        if sen_with_index[i - pair_num][0] != sen_with_index[j - pair_num][0]:
                            new.append((sen_with_index[i - pair_num][0], sen_with_index[j - pair_num][0], 0))
                            # new.append((sen_with_index[j - pair_num][0], sen_with_index[i - pair_num][0], 0))
                if sen_with_index[i][1] == 0 and sen_with_index[j][1] == 1:
                    if sen_with_index[i][2] == 0 and sen_with_index[j][2] == 0:
                        # 两句都是句子对中的第一句
                        if sen_with_index[i + pair_num][0] != sen_with_index[j + pair_num][0]:
                            new.append((sen_with_index[i + pair_num][0], sen_with_index[j + pair_num][0], 0))
                            # new.append((sen_with_index[j + pair_num][0], sen_with_index[i + pair_num][0], 0))
                    elif sen_with_index[i][2] == 0 and sen_with_index[j][2] == 1:
                        # 第一句与第二句
                        if sen_with_index[i + pair_num][0] != sen_with_index[j - pair_num][0]:
                            new.append((sen_with_index[i + pair_num][0], sen_with_index[j - pair_num][0], 0))
                            # new.append((sen_with_index[j - pair_num][0], sen_with_index[i + pair_num][0], 0))
                    elif sen_with_index[i][2] == 1 and sen_with_index[j][2] == 1:
                        # 都是第二句
                        if sen_with_index[i - pair_num][0] != sen_with_index[j - pair_num][0]:
                            new.append((sen_with_index[i - pair_num][0], sen_with_index[j - pair_num][0], 0))
                            # new.append((sen_with_index[j - pair_num][0], sen_with_index[i - pair_num][0], 0))
        nums = len(new)
        print(nums)
        if nums > 8000:
            break

    print(len(new))
    new = set(new)
    print(len(new))

    with open('./oppo_breeno_round1_data/new_valid_negative_data.tsv', "w+") as f:
        for pair in new:
            f.write("{}\t{}\t{}\n".format(pair[0], pair[1], int(pair[2])))


def add_instance2train_data():
    data = []
    with open('./oppo_breeno_round1_data/new_valid_positive_data.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            data.append((a, b, c))
    with open('./oppo_breeno_round1_data/new_valid_negative_data.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            data.append((a, b, c))
    print(len(data))
    print(len(set(data)))
    with open('./oppo_breeno_round1_data/new_valid_data.tsv', "w+") as f:
        for pair in data:
            f.write("{}\t{}\t{}\n".format(pair[0], pair[1], int(pair[2])))


def extract_positive_instance():
    pos_pair = []
    with open('./oppo_breeno_round1_data/train_data.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            if c == 1:
                pos_pair.append((a, b, c))
    print(len(pos_pair))

    with open('./oppo_breeno_round1_data/origin_positive_data.tsv', "w+") as f:
        for pair in pos_pair:
            f.write("{}\t{}\t{}\n".format(pair[0], pair[1], int(pair[2])))


def transform2csv():
    q1, q2, label = [], [], []
    with open('./oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv') as f:
        for l in f:
            l = l.strip().split('\t')
            assert len(l) == 3
            a, b, c = l[0], l[1], int(l[2])
            a = [str(i) for i in a.split(' ')]
            b = [str(i) for i in b.split(' ')]
            truncate_sequences(config.maxlen, -1, a, b)
            a = ' '.join(a)
            b = ' '.join(b)
            q1.append(a)
            q2.append(b)
            label.append(c)

    train_pd = pd.DataFrame({
        'q1': q1,
        'q2': q2,
        'label': label
    })
    train_pd.to_csv('./oppo_breeno_round1_data/train.csv', sep=',', index=False)

    # test_q1, test_q2 = [], []
    # with open('./oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv') as f:
    #     for l in f:
    #         l = l.strip().split('\t')
    #         assert len(l) == 2
    #         test_q1.append(l[0])
    #         test_q2.append(l[1])
    # test_pd = pd.DataFrame({
    #     'q1': test_q1,
    #     'q2': test_q2,
    # })
    # test_pd.to_csv('./oppo_breeno_round1_data/test.csv', sep=',', index=False)


def save_keep_tokens():
    min_count = 2
    # 加载数据集
    train_data1 = load_data(
        './tcdata/gaiic_track3_round1_train_20210228.tsv'
    )
    train_data2 = load_data(
        './tcdata/gaiic_track3_round2_train_20210407.tsv'
    )
    test_dataA = load_data(
        './tcdata/gaiic_track3_round1_testA_20210228.tsv'
    )
    test_dataB = load_data(
        './tcdata/gaiic_track3_round1_testB_20210317.tsv'
    )

    # 统计词频
    tokens = {}
    for d in train_data1 + train_data2 + test_dataA + test_dataB:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    tokens = {i: j for i, j in tokens.items() if j >= min_count}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    tokens = tokens[:21123]
    # tokens = [(12, 97041), (29, 86553), (19, 50130), (23, 33144)...]
    tokens = {
        t[0]: i + 5
        for i, t in enumerate(tokens)
    }  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask
    print(len(tokens))
    with open('./dataset/keep_tokens/tokens_dict_3.pkl', 'wb') as f:
        pickle.dump(tokens, f, pickle.HIGHEST_PROTOCOL)

    # BERT词频
    counts = json.load(open('./counts.json'))
    del counts['[CLS]']
    del counts['[SEP]']
    token_dict = load_vocab('./nezha-cn-base/vocab.txt')
    del token_dict['[PAD]']
    del token_dict['[UNK]']
    del token_dict['[CLS]']
    del token_dict['[SEP]']
    del token_dict['[MASK]']
    freqs = [
        counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    keep_tokens = list(np.argsort(freqs)[::-1])
    print(len(set(keep_tokens)))
    save_tokens = [0, 100, 101, 102, 103] + keep_tokens[:len(tokens)]
    print(len(save_tokens))
    with open('./dataset/keep_tokens/keep_tokens_3.txt', "w+") as f:
        for token in save_tokens:
            f.write("{}\n".format(int(token)))


if __name__ == '__main__':
    save_keep_tokens()
