English|[简体中文](https://github.com/baidu/Senta/blob/master/README.md)

# <p align=center>`Senta`</p>

`Senta` is a python library for many sentiment analysis tasks. It contains support for running multiple tasks such as sentence-level sentiment classification, aspect-level sentiment classification and opinion role labeling. The bulk of the code in this repository is used to implement [SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis](https://www.aclweb.org/anthology/2020.acl-main.374.pdf). In the paper, we demonstrate how to integrate sentiment knowledge into pre-trained models to learn a unified sentiment representation for multiple sentiment analysis tasks.

## How to use

### Pip

You can directly use the Python package to predict sentiment analysis tasks by loading a pre-trained `SKEP` model.

#### Installation

1. `Senta` supports Python 3.6 or later. This repository requires PaddlePaddle 1.6.3, please see [here](https://www.paddlepaddle.org.cn/documentation/docs/en/1.6/beginners_guide/install/index_en.html) for installaton instruction.

2. Install `Senta`

    ```shell
    python -m pip install Senta
    ```
   or

    ```shell
    git clone https://github.com/baidu/Senta.git
    cd Senta
    python -m pip install .
    ```

#### Quick Tour

    ```python
    from senta import Senta
    my_senta = Senta()
    
    # get pre-trained model, we provide three pre-trained models, all of which are based on the SKEP
    print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]
                                        # ernie_1.0_skep_large_ch, skep Chinese pre-trained model based on ERNIE 1.0 large.
                                        # ernie_2.0_skep_large_en, skep English pre-trained model based on ERNIE 2.0 large.
                                        # roberta_skep_large_en, skep English pre-trained model based on RoBERTa large, which is used in our paper.
    
    # get supported task
    print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]
    
    use_cuda = True # set True or False
    
    # predict different tasks
    my_senta.init_model(model_class="roberta_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
    texts = ["a sometimes tedious film ."]
    result = my_senta.predict(texts)
    print(result)
    
    my_senta.init_model(model_class="roberta_skep_large_en", task="aspect_sentiment_classify", use_cuda=use_cuda)
    texts = ["I love the operating system and the preloaded software."]
    aspects = ["operating system"]
    result = my_senta.predict(texts, aspects)
    print(result)
    
    my_senta.init_model(model_class="roberta_skep_large_en", task="extraction", use_cuda=use_cuda)
    texts = ["The JCC would be very pleased to welcome your organization as a corporate sponsor ."]
    result = my_senta.predict(texts)
    print(result)
    ```

### From source

You can use the source code to run pre-training and fine-tuning tasks. The `config` folder has different files to help you reproduce the results of our paper.

#### Preparation

    ```shell
    # download code
    git clone https://github.com/baidu/Senta.git
    
    # download a pre-trained skep model
    cd ./Senta/model_files
    sh download_roberta_skep_large_en.sh # download roberta_skep_large_en model. For other pre-trained skep models, you can find them in this dir.
    cd -
    
    # download task dataset
    cd ./Senta/data/
    sh download_en_data.sh # download English dataset used in our paper. For Chinese dataset, you can find its download script in this dir.
    cd - 
    ```

#### Installation

1. `Senta` supports Python 3.6 or later. This repository requires PaddlePaddle 1.6.3, please see [here](https://www.paddlepaddle.org.cn/documentation/docs/en/1.6/beginners_guide/install/index_en.html) for installaton instruction.

2. Install python dependencies

    ```shell
    python -m pip install -r requirements.txt
    ```

3. Set up environment variables such as Python, CUDA, cuDNN, PaddlePaddle in `env.sh` file. Details about environment variables related to PaddlePaddle can be found at the [PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/1.6/flags_en.html).

#### Quick Tour

1. Training
   
    ```shell
    sh ./script/run_pretrain_roberta_skep_large_en.sh # pre-trained model roberta_skep_large_en, which is used in our paper
    ```

2. Fine-tuning and predict

    ```shell 
    sh ./script/run_train.sh ./config/roberta_skep_large_en.SST-2.cls.json # fine-tuning on SST-2
    sh ./script/run_infer.sh ./config/roberta_skep_large_en.SST-2.infer.json # predict
    
    sh ./script/run_train.sh ./config/roberta_skep_large_en.absa_laptops.cls.json # fine-tuning on ABSA(laptops)
    sh ./script/run_infer.sh ./config/roberta_skep_large_en.absa_laptops.infer.json # predict
    
    sh ./script/run_train.sh ./config/roberta_skep_large_en.MPQA.orl.json # fine-tuning on MPQA 2.0
    sh ./script/run_infer.sh ./config/roberta_skep_large_en.MPQA.infer.json # predict
    ```
    
3. An old version of `Senta` can be found at [here](https://github.com/baidu/Senta/tree/v1), which includes BoW, CNN and BiLSTM models for Chinese sentence-level sentiment classification.


## Citation

If you extend or use this work, please cite the [paper](https://www.aclweb.org/anthology/2020.acl-main.374.pdf) where it was introduced:

```text
@inproceedings{tian-etal-2020-skep,
    title = "{SKEP}: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis",
    author = "Tian, Hao  and
      Gao, Can  and
      Xiao, Xinyan  and
      Liu, Hao  and
      He, Bolei  and
      Wu, Hua  and
      Wang, Haifeng  and
      wu, feng",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.374",
    pages = "4067--4076",
    abstract = "Recently, sentiment analysis has seen remarkable advance with the help of pre-training approaches. However, sentiment knowledge, such as sentiment words and aspect-sentiment pairs, is ignored in the process of pre-training, despite the fact that they are widely used in traditional sentiment analysis approaches. In this paper, we introduce Sentiment Knowledge Enhanced Pre-training (SKEP) in order to learn a unified sentiment representation for multiple sentiment analysis tasks. With the help of automatically-mined knowledge, SKEP conducts sentiment masking and constructs three sentiment knowledge prediction objectives, so as to embed sentiment information at the word, polarity and aspect level into pre-trained sentiment representation. In particular, the prediction of aspect-sentiment pairs is converted into multi-label classification, aiming to capture the dependency between words in a pair. Experiments on three kinds of sentiment tasks show that SKEP significantly outperforms strong pre-training baseline, and achieves new state-of-the-art results on most of the test datasets. We release our code at https://github.com/baidu/Senta.",
}
```
    