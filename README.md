# 情感分析

## 目录
- [简介](#简介)
- [SKEP](#SKEP)
- [代码结构](#代码结构)
- [一键化工具](#一键化工具)
- [详细使用说明](#详细使用说明)
- [Demo数据集说明](#Demo数据集说明)
- [论文效果复现](#论文效果复现)
- [文献引用](#文献引用)


## 简介
情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。它包含各式各样的任务，比如句子级情感分类、评价对象级情感分类、观点抽取、情绪分类等。情感分析是人工智能的重要研究方向，具有很高的学术价值。同时，情感分析在消费决策、舆情分析、个性化推荐等领域均有重要的应用，具有很高的商业价值。

近日，百度正式发布情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。

论文地址：https://arxiv.org/abs/2005.05635

为了方便研发人员和商业合作伙伴共享效果领先的情感分析技术，本次百度在Senta中开源了基于SKEP的情感预训练代码和中英情感预训练模型。而且，为了进一步降低用户的使用门槛，百度在SKEP开源项目中集成了面向产业化的一键式情感分析预测工具。用户只需要几行代码即可实现基于SKEP的情感预训练以及模型预测功能。

## SKEP
SKEP是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。

百度研究团队在三个典型情感分析任务，句子级情感分类（Sentence-level Sentiment Classification），评价对象级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Role Labeling），共计14个中英文数据上进一步验证了情感预训练模型SKEP的效果。实验表明，以通用预训练模型ERNIE（内部版本）作为初始化，SKEP相比ERNIE平均提升约1.2%，并且较原SOTA平均提升约2%，具体效果如下表：

<table>
    <tr>
        <td><strong><center>任务</strong></td>
        <td><strong><center>数据集合</strong></td>
        <td><strong><center>语言</strong></td>
        <td><strong><center>指标</strong></td>
        <td><strong><center>原SOTA</strong></td>
        <td><strong><center>SKEP</strong></td>
        <td><strong><center>数据集地址</strong></td>
    </tr>
    <tr>
        <td rowspan="4"><center>句子级情感<br /><center>分类</td>
        <td><center>SST-2</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>97.50</td>
        <td><center>97.60</td>
        <td><center><a href="https://gluebenchmark.com/tasks" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>Amazon-2</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>97.37</td>
        <td><center>97.61</td>
        <td><center><a href="https://www.kaggle.com/bittlingmayer/amazonreviews/data#" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>ChnSentiCorp</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>95.80</td>
        <td><center>96.50</td>
        <td><center><a href="https://ernie.bj.bcebos.com/task_data_zh.tgz" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>NLPCC2014-SC</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>78.72</td>
        <td><center>83.53</td>
        <td><center><a href="https://github.com/qweraqq/NLPCC2014_sentiment" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="5"><center>评价对象级的<br /><center>情感分类</td>
        <td><center>Sem-L</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>81.35</td>
        <td><center>81.62</td>
        <td><center><a href="http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>Sem-R</td>
        <td><center>英文</td>
        <td><center>ACC</td>
        <td><center>87.89</td>
        <td><center>88.36</td>
        <td><center><a href="http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>AI-challenge</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>72.87</td>
        <td><center>72.90</td>
        <td><center>暂未开放</td>
    </tr>
    <tr>
        <td><center>SE-ABSA16_PHNS</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>79.58</td>
        <td><center>82.91</td>
        <td><center><a href="http://alt.qcri.org/semeval2016/task5/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>SE-ABSA16_CAME</td>
        <td><center>中文</td>
        <td><center>ACC</td>
        <td><center>87.11</td>
        <td><center>90.06</td>
        <td><center><a href="http://alt.qcri.org/semeval2016/task5/" >下载地址</a></td>
    </tr>
    <tr>
        <td rowspan="5"><center>观点<br /><center>抽取</td>
        <td><center>MPQA-H</td>
        <td><center>英文</td>
        <td><center>b-F1/p-F1</td>
        <td><center>83.67/77.12</td>
        <td><center>86.32/81.11</td>
        <td><center><a href="https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>MPQA-T</td>
        <td><center>英文</td>
        <td><center>b-F1/p-F1</td>
        <td><center>81.59/73.16</td>
        <td><center>83.67/77.53</td>
        <td><center><a href="https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_BD</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>82.17</td>
        <td><center>84.50</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_MFW</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>86.18</td>
        <td><center>87.90</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
    <tr>
        <td><center>COTE_DP</td>
        <td><center>中文</td>
        <td><center>F1</td>
        <td><center>84.33</td>
        <td><center>86.30</td>
        <td><center><a href="https://github.com/lsvih/chinese-customer-review" >下载地址</a></td>
    </tr>
</table>

## 代码结构

```text
.
├── README.md
├── requirements.txt
├── senta                    # senta核心代码，包括模型、输出reader、分词方法等
├── script                   # 情感分析各任务入口启动脚本，通过调用配置文件完成模型的训练和预测
├── config                   # 任务配置文件目录，在配置文件中设定模型的方法、超参数、数据等
```

## 一键化工具

为了降低用户的使用门槛，百度在SKEP开源项目中集成了面向产业化的一键式情感分析预测工具。具体安装和使用方法如下：

### 安装方法

本仓库支持pip安装和源码安装两种方式，使用pip或者源码安装时需要先安装PaddlePaddle，PaddlePaddle安装请参考[安装文档](https://www.paddlepaddle.org.cn/install/quick)。

1. pip安装
```shell
python -m pip install Senta
```

2. 源码安装
```shell
git clone https://github.com/baidu/Senta.git .
cd Senta
python -m pip install .
```

### 使用方法
```python
from senta import Senta

my_senta = Senta()

# 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

# 获取目前支持的预测任务
print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]

# 选择是否使用gpu
use_cuda = True # 设置True or False

# 预测中文句子级情感分类任务
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)
texts = ["中山大学是岭南第一学府"]
result = my_senta.predict(texts)
print(result)

# 预测中文评价对象级的情感分类任务
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="aspect_sentiment_classify", use_cuda=use_cuda)
texts = ["百度是一家高科技公司"]
aspects = ["百度"]
result = my_senta.predict(texts, aspects)
print(result)

# 预测中文观点抽取任务
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="extraction", use_cuda=use_cuda)
texts = ["唐 家 三 少 ， 本 名 张 威 。"]
result = my_senta.predict(texts, aspects)
print(result)

# 预测英文句子级情感分类任务（基于SKEP-ERNIE2.0模型）
my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
texts = ["a sometimes tedious film ."]
result = my_senta.predict(texts)
print(result)

# 预测英文评价对象级的情感分类任务（基于SKEP-ERNIE2.0模型）
my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)
texts = ["I love the operating system and the preloaded software."]
aspects = ["operating system"]
result = my_senta.predict(texts, aspects)
print(result)

# 预测英文观点抽取任务（基于SKEP-ERNIE2.0模型）
my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="extraction", use_cuda=use_cuda)
texts = ["The JCC would be very pleased to welcome your organization as a corporate sponsor ."]
result = my_senta.predict(texts)
print(result)

# 预测英文句子级情感分类任务（基于SKEP-RoBERTa模型）
my_senta.init_model(model_class="roberta_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
texts = ["a sometimes tedious film ."]
result = my_senta.predict(texts)
print(result)

# 预测英文评价对象级的情感分类任务（基于SKEP-RoBERTa模型）
my_senta.init_model(model_class="roberta_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)
texts = ["I love the operating system and the preloaded software."]
aspects = ["operating system"]
result = my_senta.predict(texts, aspects)
print(result)

# 预测英文观点抽取任务（基于SKEP-RoBERTa模型）
my_senta.init_model(model_class="roberta_skep_large_en", task="extraction", use_cuda=use_cuda)
texts = ["The JCC would be very pleased to welcome your organization as a corporate sponsor ."]
result = my_senta.predict(texts)
print(result)
```

## 详细使用说明

### 项目下载

1. 代码下载
    
    下载代码库到本地
    ```shell
    git clone https://github.com/baidu/Senta.git .
    ```

2. 模型下载
   
    下载情感分析预训练SKEP的中文模型和英文模型（本项目中开放了以[ERNIE 1.0 large(中文)](https://github.com/PaddlePaddle/ERNIE)、[ERNIE 2.0 large(英文)](https://github.com/PaddlePaddle/ERNIE)和[RoBERTa large(英文)](https://github.com/pytorch/fairseq/tree/master/examples/roberta)作为初始化，训练的中英文情感预训练模型）
    ```shell
    cd ./model_files

    # 以ERNIE 1.0 large(中文)作为初始化，训练的SKEP中文情感预训练模型（简写为SKEP-ERNIE1.0）
    sh download_ernie_1.0_skep_large_ch.sh
    
    # 以ERNIE 2.0 large(英文)作为初始化，训练的SKEP英文情感预训练模型（简写为SKEP-ERNIE2.0）
    sh download_ernie_2.0_skep_large_en.sh

    # 以RoBERTa large(英文)作为初始化，训练的SKEP英文情感预训练模型（简写为SKEP-RoBERTa）
    sh download_roberta_skep_large_en.sh
    ```

3. demo数据下载
   
    下载demo数据用作SKEP训练和情感分析任务训练
    ```shell
    cd ./data/
    sh download_ch_data.sh # 中文测试数据
    sh download_en_data.sh # 英文测试数据
    ```

### 环境安装
1. PaddlePaddle 安装
    
    本项目依赖于 PaddlePaddle 1.6.3，PaddlePaddle安装后，需要及时的将 CUDA、cuDNN、NCCL2 等动态库路径加入到环境变量 LD_LIBRARY_PATH 之中，否则训练过程中会报相关的库错误。具体的paddlepaddle配置细节请查阅这里 [安装文档](https://www.paddlepaddle.org.cn/install/quick)。
    
    推荐使用pip安装方式
    ```shell
    python -m pip install paddlepaddle-gpu==1.6.3.post107 -i https://mirror.baidu.com/pypi/simple
    ```

2. senta项目python包依赖

    支持Python 3 的版本要求 3.7；
    项目中其他python包依赖列在根目录下的requirements.txt文件中，使用以下命令安装:
    ```shell
    python -m pip install -r requirements.txt
    ```

3. 环境变量添加
    
    在./env.sh中修改环境变量，包括python、CUDA、cuDNN、NCCL2、PaddlePaddle相关环境变量，PaddlePaddle环境变量说明请参考 [PaddlePaddle环境变量说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/flags_cn.html)

### 模型训练和预测

1. Pre-train训练
   
    ```shell
    #  在SKEP-ERNIE1.0中文模型的基础上，继续pre-train
    sh ./src/run_pretrain_ernie_1.0_skep_large_ch.sh

    # 在SKEP-ERNIE2.0英文模型的基础上，继续pre-train
    sh ./src/run_pretrain_ernie_2.0_skep_large_en.sh

    # 在SKEP-RoBERTa英文模型的基础上，继续pre-train
    sh ./src/run_pretrain_roberta_skep_large_en.sh
    ```

2. Finetune训练和预测句子级情感分类任务
    
    ```shell
    # 基于SEKP-ERNIE1.0模型finetune训练和预测中文句子级情感分类任务，示例数据：ChnSentiCorp
    sh ./src/run_train.sh ./config/ernie_1.0_skep_large_ch.Chnsenticorp.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_1.0_skep_large_ch.Chnsenticorp.infer.json # 预测
    ```
    
    ```shell
    # 基于SKEP-ERNIE2.0模型finetune训练和预测英文句子级情感分类任务，示例数据：SST-2
    sh ./src/run_train.sh ./config/ernie_2.0_skep_large_en.SST-2.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_2.0_skep_large_en.SST-2.infer.json # 预测
    ```
    
    ```shell 
    # 基于SKEP-RoBERTa模型finetune训练和预测英文句子级情感分类任务，示例数据：SST-2
    sh ./src/run_train.sh ./config/roberta_skep_large_en.SST-2.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/roberta_skep_large_en.SST-2.infer.json # 预测
    ```

3. Finetune训练和预测评价对象级的情感分类任务
    
    ```shell
    # 基于SKEP-ERNIE1.0模型finetune训练和预测中文评价对象级的情感分类任务，示例数据：SE-ABSA 16_PHNS
    sh ./src/run_train.sh ./config/ernie_1.0_skep_large_ch.SE-ABSA16_PHNS.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_1.0_skep_large_ch.SE-ABSA16_PHNS.infer.json # 预测
    ```
    
    ```shell 
    # 基于SEKP-ERNIE2.0模型finetune训练和预测英文评价对象级的情感分类任务，示例数据：Sem-L
    sh ./src/run_train.sh ./config/ernie_2.0_skep_large_en.absa_laptops.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_2.0_skep_large_en.absa_laptops.infer.json # 预测
    ```
    
    ```shell 
    # 基于SKEP-RoBERTa模型finetune训练和预测英文评价对象级的情感分类任务，示例数据：Sem-L
    sh ./src/run_train.sh ./config/roberta_skep_large_en.absa_laptops.cls.json # finetune训练
    sh ./src/run_infer.sh ./config/roberta_skep_large_en.absa_laptops.infer.json # 预测
    ```

4. Finetune训练和预测观点抽取或标注任务
    
    ```shell
    # 基于SKEP-ERNIE1.0模型finetune训练和预测中文观点抽取任务，示例数据：COTE_BD
    sh ./src/run_train.sh ./config/ernie_1.0_skep_large_ch.COTE_BD.oe.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_1.0_skep_large_ch.COTE_BD.infer.json # 预测
    ```
    
    ```shell
    # 基于SKEP-ERNIE2.0模型finetune训练和预测英文观点抽取任务，示例数据：MPQA 
    sh ./src/run_train.sh ./config/ernie_2.0_skep_large_en.MPQA.orl.json # finetune训练
    sh ./src/run_infer.sh ./config/ernie_2.0_skep_large_en.MPQA.infer.json # 预测
    ```
    
    ```shell 
    # 基于SKEP-RoBERTa模型finetune训练和预测英文观点抽取任务，示例数据：MPQA
    sh ./src/run_train.sh ./config/roberta_skep_large_en.MPQA.orl.json # finetune训练
    sh ./src/run_infer.sh ./config/roberta_skep_large_en.MPQA.infer.json # 预测
    ```

5. 该代码同时支持用户进一步开发使用，可以根据配置文件中设置相关数据、模型、优化器，以及修改模型的超参数进行二次开发训练。

6. 本代码库目前仅支持基于SKEP情感预训练模型进行训练和预测，如果用户希望使用Bow、CNN、LSTM等轻量级模型，请移步至[Senta v1](https://github.com/baidu/Senta/tree/v1)使用。


## Demo数据集说明
该项目中使用的各数据集的说明、下载方法及使用样例如下：

1. 句子级情感分类数据集
    
    ChnSentiCorp是中文句子级情感分类数据集，包含酒店、笔记本电脑和书籍的网购评论。为方便使用demo数据中提供了完整数据，数据示例：
    ```text
     qid	label	text_a
     0	1	這間酒店環境和服務態度亦算不錯,但房間空間太小~~不宣容納太大件行李~~且房間格調還可以~~ 中餐廳的廣東點心不太好吃~~要改善之~~~~但算價錢平宜~~可接受~~ 西餐廳格調都很好~~但吃的味道一般且令人等得太耐了~~要改善之~~
     1	<荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!
     2	0	商品的不足暂时还没发现，京东的订单处理速度实在.......周二就打包完成，周五才发货...
     ...
     ```
    
    SST-2是英文句子情感分类数据集，主要由电影评论构成。为方便使用demo数据中提供了完整数据，数据集[下载地址](https://gluebenchmark.com/tasks)，数据示例：
    ```text
    qid	label	text_a
    0	1	it 's a charming and often affecting journey .
    1	0	unflinchingly bleak and desperate
    2	1	allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker .
    ...
    ```

2. 评价对象级情感分类数据集

    SE-ABSA16_PHNS是中文评价对象级情感分类数据集，主要由描述手机类别某个属性的商品用户评论构成。为方便使用demo数据中提供了完整数据，数据集[下载地址](http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-mobile-phones-reviews-chinese-train-data-subtask-1/f651041268d411e59f7c842b2b6a04d77f78a1885b994740895c77b3fd15c69a/)，数据集示例如下:
    ```text
    qid	label	text_a	text_b
    0	1	software#usability	刚刚入手8600，体会。刚刚从淘宝购买，1635元（包邮）。1、全新，应该是欧版机，配件也是正品全新。2、在三星官网下载了KIES，可用免费软件非常多，绝对够用。3、不到2000元能买到此种手机，知足了。
    1	1	display#quality	mk16i用后的体验感觉不错，就是有点厚，屏幕分辨率高，运行流畅，就是不知道能不能刷4.0的系统啊
    2	1	phone#operation_performance	mk16i用后的体验感觉不错，就是有点厚，屏幕分辨率高，运行流畅，就是不知道能不能刷4.0的系统啊
    ...
    ```

    Sem-L数据集是英文评价对象级情感分类数据集，主要由描述笔记本电脑类别某个属性的商品用户评论构成。为方便使用demo数据中提供了完整数据，数据集[下载地址](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)，数据集示例如下：
    ```text
    qid	text_a	text_b	label
    0	Boot time	Boot time is super fast, around anywhere from 35 seconds to 1 minute.	0
    1	tech support	tech support would not fix the problem unless I bought your plan for $150 plus.	1
    2	Set up	Set up was easy.	0
    3	Windows 8	Did not enjoy the new Windows 8 and touchscreen functions.	1
    ...
    ```

3. 观点抽取抽取数据集
    
    COTE-BD数据集是中文互联网评论数据集。为方便使用demo数据中提供了完整数据，数据集[下载地址](https://github.com/lsvih/chinese-customer-review)，数据集使用例子如下，其中为了方便模型使用，下面数据是将文本进行分词处理后结果，标签用BIO标记评论实体或者事件。
    ```text
    ...
    B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O	张 莉 ， 女 ， 祖 籍 四 川 ， 1982 年 考 入 西 安 美 术 学 院 工 艺 系 ， 1986 留 校 任 教 至 今 。
    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O O O O O O O O O O O O O O O O	可 能 本 片 确 实 应 该 在 电 影 院 看 3d ， 才 能 体 会 到 奥 斯 卡 对 其 那 么 多 技 术 的 表 扬 ， 也 才 能 体 会 到 马 丁 想 用 技 术 的 进 步 对 老 电 影 致 敬 的 用 意 [UNK] 最 近 听 说 《 雨 果 》 五 月 国 内 排 片 ， 想 说 ： 广 电 搞 毛 啊 ！ 。
    O B I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O	《 笑 忘 书 》 是 由 林 夕 作 词 ， c . y . kong 作 曲 ， 王 菲 演 唱 的 一 首 歌 ， 收 录 于 专 辑 《 寓 言 》 中 。
    B I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O	龙 泉 寺 中 精 致 的 壁 画 ， 近 前 观 看 每 位 人 物 面 部 表 情 都 表 现 得 栩 栩 如 生 ， 文 革 中 部 分 被 损 坏 后 来 修 复 。
    ...
    ```
    
    MPQA数据集是英文互联网评论数据集。为方便使用demo数据中提供了完整数据，数据集[下载地址](https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/)，数据集使用例子如下，其中为了方便模型使用需要将文本进行分词处理，标签用BIO标记评论内容、评论实体和实体内容表达主体。
    ```text
    ...
    O O O B_H B_DS B_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T I_T O O O O O	In most cases he described the legal punishments like floggings and executions of murderers and major drug traffickers that are applied based on the Shria , or Islamic law as human rights violations .
    O O O O B_H B_DS I_DS I_DS I_DS I_DS B_T O O O O O O O	In other cases , he made unfounded charges by accusing Iran of discrimination against women and minorities .
    B_H B_DS I_DS I_DS O O O O O O O O O O O O O O O O O O O O	He made such charges despite the fact that women 's political , social and cultural participation is not less than that of men .
    O O O B_H B_DS O O O O O B_T I_T I_T I_T I_T I_T I_T I_T I_T O O O O O O O O O O O O O	For instance , he denounced as a human rights violation the banning and seizure of satellite dishes in Iran , while the measure has been taken in line with the law .
    ...
    ```

## 论文效果复现

基于该项目可以实现对于论文 Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis 效果的复现。下面给出论文效果的复现方法示例：

```shell 
#下载以Roberta作为初始化，训练的SKEP英文情感预训练模型（简写为SKEP-RoBERTa）
sh download_roberta_skep_large_en.sh

#基于SKEP-RoBERTa模型finetune训练和预测英文句子级情感分类任务（示例数据：SST-2）
sh ./src/run_train.sh ./config/roberta_skep_large_en.SST-2.cls.json # finetune训练
sh ./src/run_infer.sh ./config/roberta_skep_large_en.SST-2.infer.json # 预测

#基于SKEP-RoBERTa模型finetune训练和预测英文评价对象级的情感分类任务（示例数据：Sem-L）
sh ./src/run_train.sh ./config/roberta_skep_large_en.absa_laptops.cls.json # finetune训练
sh ./src/run_infer.sh ./config/roberta_skep_large_en.absa_laptops.infer.json # 预测

#基于SKEP-RoBERTa模型finetune训练和预测英文观点抽取任务（示例数据：MPQA）
sh ./src/run_train.sh ./config/roberta_skep_large_en.MPQA.orl.json # finetune训练
sh ./src/run_infer.sh ./config/roberta_skep_large_en.MPQA.infer.json # 预测
```

注：如需要复现论文数据集结果，请参考论文修改对应任务的参数设置。

## 文献引用

如需使用该项目中的代码、模型或是方法，请在相关文档、论文中引用我们的工作。

```text
@misc{tian2020skep,
    title={SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis},
    author={Hao Tian and Can Gao and Xinyan Xiao and Hao Liu and Bolei He and Hua Wu and Haifeng Wang and Feng Wu},
    year={2020},
    eprint={2005.05635},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
