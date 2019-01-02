# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:46:32 2018

@author: shen1994
"""

import pickle
import codecs
import numpy as np             

if __name__ == "__main__":

    word_to_lexic, word_to_label = {}, {}
    dict_word2label, dict_label2word = {}, {}
    dict_albet2text = {}
    lexicons_bag = set()
    with codecs.open('speechs/lexicon.txt', 'r', 'utf-8') as fin:
        for line in fin:
            one_line = line.strip().split()
            word_to_lexic[one_line[0]] = one_line[1:]
            for one in one_line[1:]:
                lexicons_bag.add(one)
            albet = ''
            for one in one_line[1:]:
                albet += one
            dict_albet2text[albet] = one_line[0].encode('utf-8')
            
    lexicons_bag.add(' ')
    for i, e in enumerate(lexicons_bag):
        dict_label2word[i] = e
        dict_word2label[e] = i
    for k in word_to_lexic.keys():
        word_to_label[k] = [dict_word2label[one] for one in word_to_lexic[k]]
    
    label_to_text, text_to_label = {}, {}    
    with codecs.open('speechs/aishell_transcript_v0.8.txt', 'r', 'utf-8') as fin:               
        for line in fin:
            one_line = line.strip().split()
            label_to_text[one_line[0]] = []
            text_to_label[one_line[0]] = []
            for one in one_line[1:]:
                label_to_text[one_line[0]].extend(word_to_lexic[one])
                label_to_text[one_line[0]].extend(' ')
                text_to_label[one_line[0]].extend(word_to_label[one])
                text_to_label[one_line[0]].extend([dict_word2label[' ']])          
    
    texts_length = list()
    for k in text_to_label.keys():
        texts_length.append(len(text_to_label[k]))
    max_length = np.max(texts_length)

    with codecs.open('speechs/word_to_lexic.pkl', 'wb') as fout:
        pickle.dump(word_to_lexic, fout, pickle.HIGHEST_PROTOCOL)
    with codecs.open('speechs/word_to_label.pkl', 'wb') as fout:
        pickle.dump(word_to_label, fout, pickle.HIGHEST_PROTOCOL)
    with codecs.open('speechs/label_to_text.pkl', 'wb') as fout:
        pickle.dump(label_to_text, fout, pickle.HIGHEST_PROTOCOL)                
    with codecs.open('speechs/text_to_label.pkl', 'wb') as fout:
        pickle.dump(text_to_label, fout, pickle.HIGHEST_PROTOCOL)  
    with codecs.open('speechs/dict_word2label.pkl', 'wb') as fout:
        pickle.dump(dict_word2label, fout, pickle.HIGHEST_PROTOCOL)                
    with codecs.open('speechs/dict_label2word.pkl', 'wb') as fout:
        pickle.dump(dict_label2word, fout, pickle.HIGHEST_PROTOCOL)

    with codecs.open('speechs/dict_albet2text.pkl', 'wb') as fout:
        pickle.dump(dict_albet2text, fout, pickle.HIGHEST_PROTOCOL)               
    with codecs.open('speechs/text_info.txt', 'w') as fout: 
        fout.write('[PICO ASR AI]\n')           
        fout.write('words: %s\n' %(str(len(dict_word2label))))
        fout.write('space encode: %s\n' %(str(dict_word2label[' '])))
        fout.write('texts: %s\n' %(str(len(text_to_label.keys()))))
        fout.write('texts max_length: %s\n' %(str(max_length)))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                