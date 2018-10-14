
# coding: utf-8

# In[2]:


import re
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer as ss
import chardet
import codecs


# In[3]:


labels={0:'G:/PYCHARM/untitled3/20news-18828/alt.atheism',
       1:'G:/PYCHARM/untitled3/20news-18828/comp.graphics',
        2:'G:/PYCHARM/untitled3/20news-18828/comp.os.ms-windows.misc',
        3:'G:/PYCHARM/untitled3/20news-18828/comp.sys.ibm.pc.hardware',
        4:'G:/PYCHARM/untitled3/20news-18828/comp.sys.mac.hardware',
        5:'G:/PYCHARM/untitled3/20news-18828/comp.windows.x',
        6:'G:/PYCHARM/untitled3/20news-18828/misc.forsale',
        7:'G:/PYCHARM/untitled3/20news-18828/rec.autos',
        8:'G:/PYCHARM/untitled3/20news-18828/rec.motorcycles',
        9:'G:/PYCHARM/untitled3/20news-18828/rec.sport.baseball',
        10:'G:/PYCHARM/untitled3/20news-18828/rec.sport.hockey',
        11:'G:/PYCHARM/untitled3/20news-18828/sci.crypt',
        12:'G:/PYCHARM/untitled3/20news-18828/sci.electronics',
        13:'G:/PYCHARM/untitled3/20news-18828/sci.med',
        14:'G:/PYCHARM/untitled3/20news-18828/sci.space',
        15:'G:/PYCHARM/untitled3/20news-18828/soc.religion.christian',
        16:'G:/PYCHARM/untitled3/20news-18828/talk.politics.guns',
        17:'G:/PYCHARM/untitled3/20news-18828/talk.politics.mideast',
        18:'G:/PYCHARM/untitled3/20news-18828/talk.politics.misc',
        19:'G:/PYCHARM/untitled3/20news-18828/talk.religion.misc'
       }


# In[4]:


path='G:/PYCHARM/untitled3/20news-18828'
packs=os.listdir(path)
result1=[]
result2=[]
for pack in packs:
     path1=path+"/"+pack
     print(path1)
     files=os.listdir(path1)
     result1.append(path1)
     for file in files:
          path2=path1+"/"+file
          result2.append(path2)

print('文档数:',len(result2))


# In[32]:


get_ipython().run_cell_magic('time', '', 'stp = stopwords.words(\'english\')\nstm = ss(\'english\')\nkeys_labels=list(labels.keys())\nwords_in_labels=[]\n\nglossary={}\n\nsymbols= [\',\',\'.\',\':\',\'_\',\'!\',\'?\',\'/\',\'\\\'\',\'\\"\',\'*\',\'>\',\'<\',\'@\',\'~\',\'-\',\'(\',\')\',\'%\',\'=\',\'\\\\\',\'^\'\n     ,\'&\',\'|\',\'#\',\'$\',\'0\',\'1\',\'2\',\'3\',\'4\',\'5\',\'6\',\'7\',\'8\',\'9\',\'10\',\'[\',\']\',\'+\',\'{\',\'}\',\';\',\'`\',\'~\']\n\nvocab=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]  #用来存放每个class文档所含的词（筛掉后）及数目\nvocab_deputy=[{},{},{},{},{},                                           #用来存储每个class文档的关键词\n    {},{},{},{},{},\n    {},{},{},{},{}, \n    {},{},{},{},{},]\nfor key in keys_labels:\n    text_path=[]\n    wordlist={}\n    for path0 in os.listdir(labels[key]):\n        path1=labels[key]+"/"+path0\n        text_path.append(path1)\n        \n    for lane in text_path:\n        d = open(lane,\'rb\')\n        data = d.read()\n        encode = chardet.detect(data)[\'encoding\']\n        with codecs.open(lane, encoding=encode) as d:\n            words = d.read()\n            for symbol in symbols:\n                 words = words.replace(symbol,\'\')\n            words = words.split()\n            for word in words:\n                word = stm.stem(word)\n                if word not in stp:\n                    if word in wordlist:\n                        wordlist[word]+=1\n                    else:\n                        wordlist[word]=1\n            \n    for i in wordlist:\n        if wordlist[i] >= 4 and wordlist[i] <= 1000:  #筛掉词频过低及过高的词\n            glossary[i]=wordlist[i]\n            vocab[key].append("%s,%s\\n" %(i,wordlist[i]))\n            vocab_deputy[key][i]=0                    #将每个class关键词存入vocab_deputy中\n    \n    with open(\'wordsbag.txt\',\'w+\',encoding=\'utf-8\') as f:\n        f.writelines(vocab[key])\n        \n        ')


# In[33]:


print(vocab_deputy[0].keys())


# In[34]:


print(type(vocab[1]))


# In[36]:


print(vocab[5])


# In[73]:


vocab_specialword_prob =  [
    {},{},{},{},{}, 
    {},{},{},{},{},
    {},{},{},{},{}, 
    {},{},{},{},{},
              ] #用来存储每个class中每个词所对应的“有该词的文档”在该类中的概率


# In[98]:


get_ipython().run_cell_magic('time', '', 'len_label=[]\ntest_before=[]\nfor key in keys_labels:\n    dic_keys=list(vocab_deputy[key].keys())\n    text_path2=[]\n    num = 0\n    for path0 in os.listdir(labels[key]):\n        path1=labels[key]+"/"+path0\n        text_path2.append(path1)\n    \n    for lane in text_path2:\n        space=[]\n        f = open(lane,\'rb\')\n        data = f.read()\n        encode = chardet.detect(data)[\'encoding\']\n        with codecs.open(lane, encoding=encode) as f:\n            lines=f.read()\n            for symbol in symbols:\n                 lines = lines.replace(symbol,\'\')\n            lines = lines.split()\n            lines = [line.strip(\'\\n\') for line in lines]\n            lines=set(lines)\n            for item in lines:\n                item = stm.stem(item)\n                if item in dic_keys:\n                    vocab_deputy[key][item]+=1   #用来存储每个class中所含有的总词数（每个词在其出现的文档中只在该文档处计算一次）\n                    space.append(item)\n            num+=len(set(space))\n            space.insert(0,key)\n            test_before.append(space)            #存入每个文档的词袋以构造样本集\n            \n            \n    len_label.append(num)                 ')


# In[99]:


import random
random.shuffle(test_before)   #打乱样本集


# In[103]:



print( vocab_deputy[2])


# In[107]:


print(len_label)   #每个class文档中出现的词所对应的出现文档数之和（一个词在一个文档中只计算一次）
print(sum(len_label))


# In[108]:


D={}    
for key in keys_labels:    
    D.update(vocab_deputy[key])   #合并20个字典


# In[109]:


print(test_before[0])


# In[110]:


DD=list(D.keys())    #包含20类文档所有关键词的list
print(DD)


# In[111]:


#此处字典初始化很重要，防止下面计算概率时出现key error
for k in vocab_specialword_prob:
    for j in DD:
        k[j]=0

for key in keys_labels:
    for j in DD:
        if j not in vocab_deputy[key]:
            vocab_deputy[key][j]=0


# In[112]:


get_ipython().run_cell_magic('time', '', 'import math\nfor j in DD:\n    for key,k in enumerate(keys_labels):\n        vocab_specialword_prob[key][j]=math.log((vocab_deputy[key][j]+1)/(len_label[key]+2),2)  #计算每个词在每个class下出现的概率\n        ')


# In[113]:


vocab_specialword_prob[2][DD[5]]


# In[118]:


predict=[]
result=[]
test=test_before[:2000]  #从随机样本集里取2000个样本
X=[T[1:] for T in test]  
Y=[T[0] for T in test]  #存放2000个样本的标签


# In[119]:


for x in X:
    prob=np.zeros(20)
    for key in keys_labels:
        for item in x:
            prob[key]+=vocab_specialword_prob[key][item]  #采用了log，固用加法计算
    predict.append(np.argmax(prob))   
    


# In[121]:


for n in range(len(X)):
    if predict[n]==Y[n]:
        result.append(1)
    else:
        result.append(0)        


# In[122]:


print('accuracy==',sum(result)/2000*100,'%')

