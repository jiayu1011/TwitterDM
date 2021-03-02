## 基于LDA主题模型和情感词典的Twitter推文主题提取及情感分析

author：jiayu-1011

github项目地址: https://github.com/Jiayu-1011/TwitterDM



在线jupyter notebook展示：

nbviewer:

https://nbviewer.jupyter.org/github/Jiayu-1011/TwitterDM/blob/master/TwitterAnalyzing.ipynb



kaggle:

https://www.kaggle.com/jiayu991011/twitteranalyzing







#### 工程目录：

###### 文件夹

>- TwitterSpider > 爬虫相关代码
>
>
>
>- dictionaries > 
>
>  degree_dictionary → 程度词典相关文件
>
>  negative_dictionary → 否定词典
>
>  sentiment_dictionary → 情感词典
>
>  
>
>- lda_model > 
>
>  lda.model... → 已经训练好的效果较好的lda模型
>
>  
>
>- lda_topics > 
>
>  dominent_topics.csv → 每篇推文对应的主题编号和对应主题构成
>
>  topics.txt → 所有主题的概率分布构成
>
>   
>
>- key_words > 
>
>  global → 全局关键词(Top20)
>
>  per_month → 逐月关键词(Top10)
>
>  
>
>- word_cloud > 
>
>  global → 全局关键词词云
>
>  per_month → 逐月关键词词云
>
>  
>
>- senti_scores >
>
>  senti_scores.csv → 所有推文对应的情感得分
>
>





###### 文件

>- TwitterAnalyzing.ipynb → 使用LDA模型和情感词典分析（代码主文件）
>
>- GetDictionaries.ipynb → 根据词典的原txt文件初步处理并修改格式为csv文件，便于后续使用
>
>- ban_words_list.txt → 关键词提取禁用单词列表
>
>- twitter_covid.xlsx → 提取的推文数据（数据文件）
>
>- requirements.txt → 把代码跑起来需要安装的第三方包列表
>
>



### 数据分析结果

###### 关键词

> 从2020.3.12 - 2021.1.23，全局Top20的关键词为：
>
> johnson 69
> russia 50
> race 45
> volunteer 43
> promise 41
> here 41
> trump 40
> take 40
> sign 39
> trial 38
> result 38
> news 37
> early 37
> ready 37
> oxford 37
> begin 36
> know 36
> candidate 35
> available 35
> first 34
>
>  
>
> 逐月的Top10关键词见文件



###### 主题

>**topic 0:**
>  "trial"   0.015
>  "people"    0.008
>  "test"    0.007
>  "take"    0.007
>  "trump"    0.006
>  "make"    0.005
>  "health"    0.005
>  "development"    0.004
>  "news"    0.004
>  "state"   0.004
>
>**topic 1:**
>  "first"   0.009
>  "trump"    0.007
>  "trial"    0.007
>  "test"    0.005
>  "people"    0.005
>  "scientist"    0.005
>  "say"    0.004
>  "could"    0.004
>  "treatment"    0.004
>  "need"   0.004
>
>**topic 2:**
>  "say"   0.010
>  "people"    0.008
>  "trump"    0.007
>  "work"    0.007
>  "trial"    0.006
>  "test"    0.006
>  "year"    0.006
>  "news"    0.006
>  "world"    0.005
>  "first"   0.005
>
>**topic 3:**
>  "say"   0.010
>  "develop"    0.007
>  "make"    0.007
>  "take"    0.006
>  "work"    0.006
>  "trial"    0.006
>  "health"    0.005
>  "people"    0.005
>  "plan"    0.005
>  "could"   0.005
>
>



###### 情感得分

>```
>-8.0~-7.5		  1          0.009%
>-7.5~-7.0		  1          0.009%
>-7.0~-6.5         1          0.009%
>-6.5~-6.0         1          0.009%
>-6.0~-5.5         1          0.009%
>-5.5~-5.0         0          0.000%
>-5.0~-4.5         0          0.000%
>-4.5~-4.0         0          0.000%
>-4.0~-3.5         1          0.009%
>-3.5~-3.0         2          0.017%
>-3.0~-2.5         3          0.026%
>-2.5~-2.0         6          0.051%
>-2.0~-1.5        44          0.375%
>-1.5~-1.0       168          1.431%
>-1.0~-0.5       641          5.460%
>-0.5~ 0.0      3431         29.225%
> 0.0 ~ 0.5      5442         46.354%
> 0.5 ~ 1.0      1688         14.378%
> 1.0 ~ 1.5       627          5.341%
> 1.5 ~ 2.0       188          1.601%
> 2.0 ~ 2.5        59          0.503%
> 2.5 ~ 3.0        13          0.111%
> 3.0 ~ 3.5         4          0.034%
> 3.5 ~ 4.0         1          0.009%
> 4.0 ~ 4.5         0          0.000%
> 4.5 ~ 5.0         0          0.000%
> 5.0 ~ 5.5         0          0.000%
> 5.5 ~ 6.0         0          0.000%
> 6.0 ~ 6.5         0          0.000%
> 6.5 ~ 7.0         0          0.000%
> 7.0 ~ 7.5         0          0.000%
> 7.5 ~ 8.0         0          0.000%
>```
>
>
>- 拥有正向情感的推文约占了64%，拥有负向情感的推文只约占了36%，可以看出还是有较多的人对covid vaccine有较积极的态度的
>- 接近60%推文的情感得分落在了(0,1)之间
>
>





#### 本地跑代码流程:

1. 安装Anaconda，配置好相关python环境（我的版本是python3.7.5）
2. 打开Anaconda Prompt（Anaconda的命令行)，cd进入工程目录`TwitterDM`后，`pip install -r requrements.txt`安装相关的第三方包
3. 安装完毕后打开jupyter notebook（装Anaconda的时候应该是有自带装有jupyter，如果没有就再`pip install jupyter`装一下）
4. 打开jupyter，run就完了



#### kaggle平台在线跑代码流程:



