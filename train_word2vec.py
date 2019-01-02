# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:51:56 2018

@author: shen1994
"""

import os
import gensim
import codecs

class Sentences:
    def __init__(self, dirname):
        self.dirname = dirname 
        self.counter = 0
        
    def __iter__(self):
        
        for line in codecs.open(self.dirname, 'r', 'utf-8'):
            pieces = line.strip().split()
            yield pieces[1:]
        self.counter += 1
        print('\ntrain_once---%03d!' %self.counter)

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
     
    if not os.path.exists("model"):
        os.makedirs("model")
    
    sentences = Sentences('speechs/aishell_transcript_v0.8.txt')
    model = gensim.models.Word2Vec(sentences, size=128, window=5, min_count=1, iter=100, workers=2)
    model.save('model/model_vector_dict.m')
    model.wv.save_word2vec_format('model/model_vector_dict.txt', binary=False)
    
    '''    
    # gensim.models.Word2Vec参数说明
    # sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    # sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # alpha: 是学习速率
    # seed：用于随机数发生器。与初始化词向量有关。
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    # sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
    # workers参数控制训练的并行数。
    # hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
    # negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
    # cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
    # hashfxn： hash函数来初始化权重。默认使用python的hash函数
    # iter： 迭代次数，默认为5
    # trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    # sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    # batch_words：每一批的传递给线程的单词的数量，默认为10000
    '''
