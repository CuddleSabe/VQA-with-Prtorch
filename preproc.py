import os
import re
import pdb
import sys
import csv
import json

import base64
import pickle
from typing import Dict, Any
import joblib
import numpy as np
import nltk

csv.field_size_limit(2147483647)
nltk.data.path.append('data')
# nltk.download('punkt', download_dir='data')
# word_tokenize 用于分词
from nltk.tokenize import word_tokenize
from tqdm import tqdm

ta_path = os.path.join('.//data', 'v2_mscoco_train2014_annotations.json')
va_path = os.path.join('data', 'v2_mscoco_val2014_annotations.json')
tq_path = os.path.join('data', 'v2_OpenEnded_mscoco_train2014_questions.json')
vq_path = os.path.join('data', 'v2_OpenEnded_mscoco_val2014_questions.json')
glove_path = os.path.join('data', 'glove', 'glove.6B.300d.txt')
# visual feature 视觉特征
vfeats_path = os.path.join('data', 'trainval_resnet101_faster_rcnn_genome_36.tsv')

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!'
]

"""
输入: inText(string)
返回: 将符号替换为空格的outText(UNICODE编码)(string)
"""
def _process_punctuation(inText):
    outText = inText
    for p in punct:
        # 将inText中的标点符号和换行符等特殊符号替换成空格
        if(p + ' ' in inText or ' ' + p in inText) \
        or (re.search(comma_strip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    # 转换为 UNICODE 编码
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

"""
输入: inText(string)
返回: 替换掉英文数字和冠词的outText(string)
"""
def _process_digit_article(inText):
    outText = []
    # 将 outText 小写并分词为列表
    tempText = inText.lower().split()
    for word in tempText:
        # 将英文数字转换为阿拉伯数字
        word = manual_map.setdefault(word, word)
        # 去掉 a, an, the 冠词
        if word not in articles:
            outText.append(word)
        else:
            pass
    # 去掉单词中的'
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    # 将 outText 连成字符串
    outText = ' '.join(outText)
    return outText

"""
输入: 频率阈值 freq_thr(int)
返回: 问题的词典(dict)
"""
def process_a(freq_thr=9):
    # ta 与 va 包含了标注的坐标、面积等信息
    ta = json.load(open(ta_path))['annotations']
    va = json.load(open(va_path))['annotations']
    annos = ta + va
    print("正在计算多选问题答案的频数...")
    mca_freqs = {}
    for anno in tqdm(annos):
        mca = _process_digit_article(_process_punctuation(anno['multiple_choice_answer']))
        mca = mca.replace(',', '')
        mca_freqs[mca] = mca_freqs.get(mca, 0) + 1

    # 去除稀有答案
    for a, freq in list(mca_freqs.items()):
        if freq < freq_thr:
            mca_freqs.pop(a)

    print("答案出现次数大于 {} 的数量为: {}".format(freq_thr - 1, len(mca_freqs)))

    # 总结答案字典
    idx2ans = []  # 所有答案按照编号排序
    ans2idx = {}  # KEY:答案 VALUE:第一次出现的序号 的字典
    for i, a in enumerate(mca_freqs):
        idx2ans.append(a)
        ans2idx[a] = i

    print("正在计算 soft scores...")
    targets = []
    for anno in tqdm(annos):
        anss = anno['answers']

        # 计算独立答案的频数
        ans_freqs = {}
        for a in anss:
            # 统计各 answer 出现的频数
            ans_freqs[a['answer']] = ans_freqs.get(a['answer'], 0) + 1

        soft_score = []
        for a, freq in ans_freqs.items():
            if a in ans2idx:
                # 这里是限制 score 范围为 0-1
                soft_score.append((a, min(1, freq / 3)))

        targets.append({
            'question_id': anno['question_id'],
            'image_id': anno['image_id'],
            'answer': soft_score  # [(ans1, score1), (ans2, score2), ...]
        })

    pickle.dump([idx2ans, ans2idx], open(os.path.join('data', 'dict_ans.pkl'), 'wb'))
    return targets

"""
输入: tbt(string)
返回: 去掉,和？的tbt(string)
"""
def _tokenize(tbt):
    tbt = tbt.lower().replace(',', '').replace('?', '')
    return word_tokenize(tbt)

# max_words=14
def process_qa(targets):
    print("正在结合问题和答案...")
    idx2word = []
    word2idx = {}
    tq = json.load(open(tq_path))['questions']
    vq = json.load(open(vq_path))['questions']
    qs = tq + vq
    qas = []

    for i, q in enumerate(tqdm(qs)):
        tokens = _tokenize(q['question'])
        for t in tokens:
            # 将所有单词放到 idx2word 中， word2idx 记录单词第一次出现的序号
            if not t in word2idx:
                idx2word.append(t)
                word2idx[t] = len(idx2word) - 1

        assert q['question_id'] == targets[i]['question_id'], \
            "Question ID doesn't match ({}: {})".format(q['question_id'], targets[i]['question_id'])

        qas.append({
            'image_id': q['image_id'],
            'question': q['question'],
            'question_id': q['question_id'],
            'question_toked': tokens,
            'answer': targets[i]['answer']
        })

    pickle.dump([idx2word, word2idx], open(os.path.join('data', 'dict_q.pkl'), 'wb'))
    pickle.dump(qas[:len(tq)], open(os.path.join('data', 'train_qa.pkl'), 'wb'))
    pickle.dump(qas[len(tq):], open(os.path.join('data', 'val_qa.pkl'), 'wb'))
    return idx2word

"""
输入: idx2word(list)
返回: 预训练权重pretrained_weights(file)
"""
def process_wemb(idx2word):
    print("正在加载 Embedding 预权重...")
    word2emb = {}
    emb_dim = int(glove_path.split('.')[-2].split('d')[0])
    # 以 50d 中的单词"the"举例，词向量格式为"the + 50个浮点数"
    with open(glove_path, encoding='UTF-8') as f:
        # 一次读取一行
        for entry in f:
            vals = entry.split(' ')
            # 提取当前行词向量的单词
            word = vals[0]
            # 将当前单词的词向量存入
            word2emb[word] = np.asarray(vals[1:], dtype=np.float32)

    # 预训练权重矩阵大小为 [单词词典数，Embedding维度]
    pretrained_weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)
    # 将 GloVe 预训练好的词向量word2emb初始化到预训练权重中去
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        pretrained_weights[idx] = word2emb[word]

    np.save(os.path.join('data', 'glove_pretrained_{}.npy'.format(emb_dim)), pretrained_weights)

def process_vfeats():
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    tq = json.load(open(tq_path))['questions']
    vq = json.load(open(vq_path))['questions']
    tids = set([q['image_id'] for q in tq])
    vids = set([q['image_id'] for q in vq])

    print("读取 tsv 文件中， 共需要迭代次数: {}".format(len(tids)+len(vids)))
    tvfeats = {}
    vvfeats = {}
    with open(vfeats_path) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for i, item in tqdm(enumerate(reader)):
            image_id = int(item['image_id'])
            feats = np.frombuffer(base64.b64decode(item['features']),
                    dtype=np.float32).reshape((int(item['num_boxes']), -1))
            
            if image_id in tids:
                tvfeats[image_id] = feats
            elif image_id in vids:
                vvfeats[image_id] = feats
            else:
                raise ValueError("Image_id: {} not in set".format(image_id))

    print("正在通过 joblib 将 tsv 转换为 pkl 文件...请耐心等待")
    joblib.dump(tvfeats, open(os.path.join('data', 'train_vfeats.pkl'), 'wb'))
    joblib.dump(vvfeats, open(os.path.join('data', 'val_vfeats.pkl'), 'wb'))


if __name__ == '__main__':
    # targets 包含了问题编号(list),图像编号,当前图像的所有问题和score的元组的列表[(ans1,score1)()()...]
    targets = process_a()
    # 返回单词分词字典
    idx2word = process_qa(targets)
    # 将 Embedding 的预训练权重保存为glove_pretrained * .npy文件
    process_wemb(idx2word)
    # 将提取到的预训练图像特征保存为 train/val_vfeats.pkl 文件
    process_vfeats()
    print("Done")