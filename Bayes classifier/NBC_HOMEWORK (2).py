
# coding: utf-8

# # 贝叶斯分类器（伯努利模型）
#   

# In[4]:


import re
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer as ss
import chardet
import codecs


# ## 给20类文本集打上标签，并遍历所有文本文件

# In[5]:


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


# In[6]:


path='G:/PYCHARM/untitled3/20news-18828'
packs=os.listdir(path)
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


#    
#    
#    仿照之前vsm的操作，对每个class下的每个文本进行去标点、stem、去停用词的操作，每处理完一个class就将其中词频过低以及过高的词都删减掉，并将处理后剩下的词存入与该class对应的vocab[key]中，同时vocab_deput用来存储每个vocab[key]中的关键词，将各个关键词的值初始化为0是为了方便之后的统计。

# In[7]:


get_ipython().run_cell_magic('time', '', 'stp = stopwords.words(\'english\')\nstm = ss(\'english\')\nkeys_labels=list(labels.keys())\n\n\nsymbols= [\',\',\'.\',\':\',\'_\',\'!\',\'?\',\'/\',\'\\\'\',\'\\"\',\'*\',\'>\',\'<\',\'@\',\'~\',\'-\',\'(\',\')\',\'%\',\'=\',\'\\\\\',\'^\'\n     ,\'&\',\'|\',\'#\',\'$\',\'0\',\'1\',\'2\',\'3\',\'4\',\'5\',\'6\',\'7\',\'8\',\'9\',\'10\',\'[\',\']\',\'+\',\'{\',\'}\',\';\',\'`\',\'~\']\n\nvocab=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]  #用来存放每个class文档所含的词\nvocab_deputy=[{},{},{},{},{},                                           #用来存储每个class文档的关键词\n    {},{},{},{},{},\n    {},{},{},{},{}, \n    {},{},{},{},{},]\nfor key in keys_labels:\n    text_path=[]\n    wordlist={}\n    for path0 in os.listdir(labels[key]):\n        path1=labels[key]+"/"+path0\n        text_path.append(path1)\n        \n    for lane in text_path:\n        d = open(lane,\'rb\')\n        data = d.read()\n        encode = chardet.detect(data)[\'encoding\']\n        with codecs.open(lane, encoding=encode) as d:\n            words = d.read()\n            for symbol in symbols:\n                 words = words.replace(symbol,\'\')\n            words = words.split()\n            for word in words:\n                word = stm.stem(word)\n                if word not in stp:\n                    if word in wordlist:\n                        wordlist[word]+=1\n                    else:\n                        wordlist[word]=1\n            \n    for i in wordlist:\n        if wordlist[i] >= 4 and wordlist[i] <= 1000:  #筛掉词频过低及过高的词\n            vocab[key].append("%s" %i)\n            vocab_deputy[key][i]=0                    #将每个class关键词存入vocab_deputy中')


# In[8]:


print(list(vocab_deputy[0].keys())[:100]) #第一类文档提取的前100个关键词


# In[9]:


print(type(vocab[0]))


# ## 基于贝叶斯概率计算构建模型

# In[42]:





# In[10]:


vocab_specialword_prob =  [
    {},{},{},{},{}, 
    {},{},{},{},{},
    {},{},{},{},{}, 
    {},{},{},{},{},
              ] #用来存储每个class中每个词所对应的“有该词的文档”在该类中的概率


# 此处同样是对每个class下的每个文档文件进行操作，将每个文档中经筛选后剩下的词构成一个集合，因为集合元素的互异性，所以那些在一个文档中出现多次的词在集合中也就只是一个元素了，这是方便计算每个class所含有的总词数的，即是为了满足“每个词在其出现的文档中只计算一次”。接下来只要每个文档的集合中有元素能在之前所构建的对应vocab[key]中找到，那么vocab_deputy[key]中相应位置的值就加一。同时，该类下每个文档的关键词都被提取组成一个样本存入样本集test_before。（最终该样本集将会有18828个样本） 在以上过程中每个样本集关键词的个数也被计算并在其对应的class下求和，其所构建的len_label中将存有20个class的每个class的总词数（每个词在出现的文档中只计算一次）。

# In[11]:



get_ipython().run_cell_magic('time', '', 'len_label=[]\ntest_before=\nfor key in keys_labels:\n    dic_keys=list(vocab_deputy[key].keys())\n    text_path2=[]\n    num = 0\n    for path0 in os.listdir(labels[key]):\n        path1=labels[key]+"/"+path0\n        text_path2.append(path1)\n    \n    for lane in text_path2:\n        space=[]\n        f = open(lane,\'rb\')\n        data = f.read()\n        encode = chardet.detect(data)[\'encoding\']\n        with codecs.open(lane, encoding=encode) as f:\n            lines=f.read()\n            for symbol in symbols:\n                 lines = lines.replace(symbol,\'\')\n            lines = lines.split()\n            lines = [line.strip(\'\\n\') for line in lines]\n            lines=set(lines)  #集合元素具有互异性\n            for item in lines:\n                item = stm.stem(item)\n                if item in vocab[key]:\n                    vocab_deputy[key][item]+=1   #用来存储每个class中所含有的总词数（每个词在其出现的文档中只在该文档处计算一次）\n                    space.append(item)\n            num+=len(set(space))\n            space.insert(0,key)                  #在每个样本首位插入该样本所对应的class\n            test_before.append(space)            #存入每个文档的词袋以构造样本集\n            \n    len_label.append(num)                 ')


# 
# 
# 这里随机打乱样本集，方便以后抽取部分作为测试。

# In[12]:


import random
random.shuffle(test_before)   #打乱样本集


# In[13]:



print( vocab_deputy[1])


# In[14]:


print(len_label)   #每个class文档中出现的词所对应的出现文档数之和（一个词在一个文档中只计算一次）
print(sum(len_label))


# In[15]:


D={}    
for key in keys_labels:    
    D.update(vocab_deputy[key])   #合并20个字典


# In[16]:


print(test_before[0])


# In[25]:


DD=list(D.keys())    #包含20类文档所有关键词的list
print(DD[:100])


# In[18]:


#此处字典初始化很重要，防止下面计算概率时出现key error
for k in vocab_specialword_prob:
    for j in DD:
        k[j]=0

for key in keys_labels:
    for j in DD:
        if j not in vocab_deputy[key]:
            vocab_deputy[key][j]=0


# 下面计算每个词在每个class下出现的概率，假定每个词的出现都是独立的，那么一个样本在该class下的概率将是其所含各个词在该class下概率的乘积，这样的话其值必然会很低，因此此处我们引入对数函数log，将概率计算由相乘变为相加。

# In[29]:


get_ipython().run_cell_magic('time', '', 'import math\nfor j in DD:\n    for key in keys_labels:\n        vocab_specialword_prob[key][j]=math.log((vocab_deputy[key][j]+1)/(len_label[key]+2),2)  #计算每个词在每个class下出现的概率\n        ')


# In[30]:


vocab_specialword_prob[2][DD[5]]


# ## 测试构建的模型

# 观察该样本在哪个class下的概率最大，就判断样本属于哪个class。然后将每个样本的判断结果和它的真实所属class进行比较，计算出由随机2000个样本所组测试集来测试的贝叶斯分类器的准确率。

# In[31]:


predict=[]
result=[]
test=test_before[:2000]  #从随机样本集里取2000个样本
X=[T[1:] for T in test]  
Y=[T[0] for T in test]  #存放2000个样本的标签


# In[32]:


for x in X:
    prob=np.zeros(20)
    for key in keys_labels:
        for item in x:
            prob[key]+=vocab_specialword_prob[key][item]  #采用了log，固用加法计算
    predict.append(np.argmax(prob))   #存储各个样本最大概率所属的class
    


# In[33]:


for n in range(len(X)):
    if predict[n]==Y[n]:
        result.append(1)
    else:
        result.append(0)        


# In[34]:


print('accuracy==',sum(result)/2000*100,'%')

