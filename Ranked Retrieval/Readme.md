Pivoted Length Normalization VSM and BM25
==========================
* 建立倒排索引，不仅记录对应词所在的tweetID，也记录该词在这个tweet中出现的次数
（tweetID,freq）
   
* 实现Pivoted Length Normalization VSM

* 实现BM25

* 构建querys.txt,并读取里面所有编好171到225共55条所有query的内容
![TIM截图20181130152551.jpg](https://i.loli.net/2018/11/30/5c00e647858e2.jpg)



* 将依据上述两种方式所得到的查询结果分别存入两个txt文件中

  txt每一行为：queryID 及其 查询到的相应 tweetID
  ![TIM截图20181130152934.jpg](https://i.loli.net/2018/11/30/5c00e6ea55082.jpg)

* 结果测试（MAP AND NDMG）
  
  BM25:

  ![TIM截图20181130153120.jpg](https://i.loli.net/2018/11/30/5c00e76ba33c0.jpg)


  Pivoted Length Normalization VSM: 
 
  ![TIM截图20181130153131.jpg](https://i.loli.net/2018/11/30/5c00e76ba4c05.jpg)
  