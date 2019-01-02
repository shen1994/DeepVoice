# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:01:55 2018

@author: shen1994
"""

import codecs
import pickle
import gensim
import numpy as np

def write_to_text_dict():
        
    dict_albet2text2 = {}
    with codecs.open('speechs/lexicon.txt', 'r', 'utf-8') as fin:
        for line in fin:
            one_line = line.strip().split()
            albet = ''
            for one in one_line[1:]:
                albet += one
            if albet not in dict_albet2text2.keys():
                dict_albet2text2[albet] = []
            text = one_line[0].encode('utf-8')
            if text not in dict_albet2text2[albet]:
                dict_albet2text2[albet].append(one_line[0].encode('utf-8'))
    
    with codecs.open('speechs/dict_albet2text2.pkl', 'wb') as fin:
        pickle.dump(dict_albet2text2, fin, pickle.HIGHEST_PROTOCOL)      

def load_text_dict():
    
    with codecs.open('speechs/dict_albet2text2.pkl', 'rb') as fout:
        dict_albet2text = pickle.load(fout)
    return dict_albet2text
    
def load_vector_dict():
    
    text_vector = gensim.models.Word2Vec.load(r'model/model_vector_dict.m')
    
    words_bag = list(text_vector.wv.vocab.keys())
    
    return text_vector, words_bag

def speech2text(syllable, front_text, text_dict, vector_dict, words_bag, max_short):

    syllable_len = len(syllable) 
    if syllable_len < 0:
        return '' 
     
    if syllable_len < max_short:
        return speech2decode(syllable, front_text, 
                             syllable_len, text_dict, 
                             vector_dict, words_bag)
       
    text = []
    pointer = 0
    
    syllable_cell = [syllable[i] for i in range(max_short)]
    for i in range(syllable_len):
        if i < pointer:
            continue
        
        # syllable to text
        if text is not None:
            cell_index, cell_text = speech2decode2(syllable_cell, text, 
                                                   max_short, text_dict, 
                                                   vector_dict, words_bag)
        else:
            cell_index, cell_text = speech2decode2(syllable_cell, front_text, 
                                                   max_short, text_dict, 
                                                   vector_dict, words_bag)
        # update slide window
        cell_counter = 0
        for j in range(cell_index, max_short):
            syllable_cell[cell_counter] = syllable_cell[j]
            cell_counter += 1
        for j in range(0, min(cell_index, syllable_len-i-max_short)):
            syllable_cell[cell_counter] = syllable[i+max_short+j]
            cell_counter += 1
        
        # load text
        text.append(cell_text)
        pointer += cell_index
  
    return text
            
def speech2decode(cell, front_text, length, text_dict, vector_dict, words_bag):
    text = []
    pointer = 0
    for i in range(length):
        if i < pointer:
            continue        
        cell_index, cell_text = speech2decode2(cell[i:], front_text,
                                               length-i, text_dict, 
                                               vector_dict, words_bag)
        text.append(cell_text)
        pointer += cell_index

    return text  
    
def calculate_distance(a, b):    
    c = a - b
    return np.sqrt(np.dot(c, c.T))
    
def select_from_similarity(cell, front_text, text_dict, vector_dict, words_bag):
    
    if len(text_dict[cell]) > 1 and len(front_text) > 0:    
        if front_text[-1] in words_bag:
            front_vector = vector_dict.wv[front_text[-1]]
                
            counter = 0
            word_index, similarity = [], []
            for one in text_dict[cell]:
                if one.decode('utf-8') in words_bag:
                    back_vector = vector_dict.wv[one.decode('utf-8')]
                    similarity.append(calculate_distance(front_vector, back_vector))
                    word_index.append(counter)
                counter += 1
            if similarity is not None:
                text = text_dict[cell][word_index[np.argmax(similarity)]].decode('utf-8')
            else:
                text = text_dict[cell][0].decode('utf-8') 
        else:
            text = text_dict[cell][0].decode('utf-8')
                    
    else:
        text = text_dict[cell][0].decode('utf-8')   
        
    return text
    
def speech2decode2(cell, front_text, length, text_dict, vector_dict, words_bag):
    
    index = 1
    text = ''
    find_from_bag = False
    for i in range(length):
        cell_str = ''
        for n in range(len(cell[:length-i])):
            cell_str += cell[n]
        if cell_str in text_dict.keys():

            find_from_bag = True
            index = length - i
            # albet has many words
            text = select_from_similarity(cell_str, front_text, text_dict,
                                          vector_dict, words_bag)   
            break
        
    # albet is wrong
    if not find_from_bag:
        pass 
    
    return index, text
    
def albet2text(albet, text_dict, vector_dict, words_bag):
     
    long_long_albet = []
    
    for one in albet.strip().split():
        
        one_albet = []
        one_str = ''
        for c in one:
            one_str += c
            if c >= '1' and c <= '5':
                one_albet.append(one_str)
                one_str = ''
                continue
        if one_str is not '':
            one_str += '5'
            one_albet.append(one_str)
            
        long_long_albet.append(one_albet)
        
    front_text = []
    for long_albet in long_long_albet:
        front_text.extend(speech2text(long_albet, front_text, text_dict, vector_dict, words_bag, 7))
    
    text = ''
    for one in front_text:
        text += one
        
    return text     
            
if __name__ == "__main__":
    
    text_dict = load_text_dict()
    text_vector_dict, text_words_bag = load_vector_dict()
    
    albet = ' zhu4zhai2 cheng2uuui2 qvan2guo2 fang2li4chan3 xiao1shou4 jin1eee2 uuui4iii1 zeng1zhang3 di1 ban4kua4'
    
    text = albet2text(albet, text_dict, text_vector_dict, text_words_bag)
    
    print(text)
