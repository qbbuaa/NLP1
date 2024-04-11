import jieba
import math
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def get_useless(list):
    with open(list, 'r', encoding='utf-8') as file:
        useless = set([line.strip() for line in file.readlines()])
    return useless

def get_texts(texts,rootDir):
    listdir = os.listdir(rootDir)
    for file in listdir:
        path = os.path.join(rootDir, file)
        if os.path.isfile(path) and os.path.splitext(file)[1].lower() == '.txt':
            with open(os.path.abspath(path), "r", encoding='ansi') as file:
                filename = os.path.basename(file.name)
                if re.search(r'[\u4e00-\u9fa5]', filename):
                    filecontext = file.read()
                    filecontext = filecontext.replace('\n','')
                    full_width_english = re.compile(r'[\uFF01-\uFF5E]+')
                    filecontext = full_width_english.sub('',filecontext)
                    texts[filename] = filecontext.replace('\u3000','')
        elif os.path.isdir(path):
            get_texts(texts, path)

def get_wordslists(texts,punctuation):
    wordslists = dict()
    for text_name,text in texts.items():
        words = jieba.lcut(text)
        words_noPunctuation = [word for word in words if word not in punctuation and word.isalpha() and not word.isascii()]
        wordslists[text_name] = words_noPunctuation
    return wordslists

def get_ranks(words):
    word_counts = Counter(words)
    words_sorted = word_counts.most_common()
    ranks = np.arange(1, len(words_sorted) + 1)
    frequencies = np.array([word[1] for word in words_sorted])
    return  ranks,frequencies

def prove_law(wordslists,stopwords):
    fullwords = []
    for words_name,words in wordslists.items():
        words_noStopwords = [word for word in words if word not in stopwords and word.isalpha() and not word.isascii()]
        fullwords = fullwords + words_noStopwords
    fullranks,fullfrequencies = get_ranks(fullwords)
    plt.loglog(fullranks,fullfrequencies, 'o', 10, 10)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('Zipf\'s law')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('齐夫定理.png')
    plt.close()

def get_tf(tf_dic,corpus):
    for i in range(len(corpus)):
        tf_dic[corpus[i]] = tf_dic.get(corpus[i], 0) + 1

def get_bigram_tf(tf_dic, corpus):
    for i in range(len(corpus)-1):
        tf_dic[(corpus[i], corpus[i+1])] = tf_dic.get((corpus[i], corpus[i+1]), 0) + 1

def get_trigram_tf(tf_dic,corpus):
    for i in range(len(corpus)-2):
        tf_dic[((corpus[i], corpus[i+1]), corpus[i+2])] = tf_dic.get(((corpus[i], corpus[i+1]), corpus[i+2]), 0) + 1

def cal_entropy(words):
   unigram_tf = dict()
   bigram_tf = dict()
   trigram_tf = dict()
   uni_entropy = 0
   bi_entropy = 0
   tri_entropy = 0
   get_tf(unigram_tf,words)
   words_len = len(words)
   for uni_word in unigram_tf.items():
       uni_entropy += -(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2)
   get_bigram_tf(bigram_tf,words)
   bigram_len = sum([dic[1] for dic in bigram_tf.items()])
   for bi_word in bigram_tf.items():
       bi_entropy += -(bi_word[1] / bigram_len) * math.log((bi_word[1] / unigram_tf[bi_word[0][0]]), 2)
   get_trigram_tf(trigram_tf,words)
   trigram_len = sum([dic[1] for dic in trigram_tf.items()])
   for tri_word in trigram_tf.items():
       tri_entropy += -(tri_word[1] / trigram_len) * math.log((tri_word[1] / bigram_tf[tri_word[0][0]]), 2)
   return uni_entropy,bi_entropy,tri_entropy

def get_fullwords(wordslist):
    fullwords = []
    for words_name, words in wordslist.items():
        fullwords.extend(words)
    return fullwords

def list_table(table,corpuslist):
    for key,value in corpuslist.items():
        uni_entropy, bi_entropy, tri_entropy = cal_entropy(value)
        table.append([key,uni_entropy, bi_entropy, tri_entropy ])

def save_table(table,title,savefig):
    plt.table(cellText=table, colLabels=['采用文本', 'N=1（比特/词）', 'N=2（比特/词）', 'N=3（比特/词）'], loc='center',
              cellLoc='center')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.axis('off')
    plt.title(title)
    plt.savefig(savefig)
    plt.close()

def get_strlists(wordslist):
    str_lists = dict()
    for words_name,words in wordslist.items():
        str_lists[words_name] = ''.join(words)
    return str_lists

def combine_str(strlist):
    combined_str = ''
    for key,str in strlist.items():
        combined_str = combined_str +str
    return combined_str


if __name__ == '__main__':

    punctuation = get_useless("cn_punctuation.txt")
    stopwords = get_useless("cn_stopwords.txt")

    texts = dict()
    get_texts(texts,'中文语料库')

    wordslist = get_wordslists(texts,punctuation)

    prove_law(wordslist, stopwords)

    table_word = []
    list_table(table_word,wordslist)
    fullwords = {'所有小说': get_fullwords(wordslist)}
    list_table(table_word,fullwords)
    save_table(table_word,'基于词的N元模型的中文信息熵','word.png')

    str_list = get_strlists(wordslist)
    table_character = []
    list_table(table_character,str_list)
    fullcharacters = {'所有小说':combine_str(str_list)}
    list_table(table_character,fullcharacters)
    save_table(table_character,'基于字的N元模型的中文信息熵','character.png')
