export CUDA_VISIBLE_DEVICES=4,5         # 声明可用的 GPU
python sentiment_classify.py \
    --model_type bilstm_net \
    --train_data_path ./data/train_data/corpus.train \
    --word_dict_path ./data/train.vocab \
    --mode train \
    --model_path ./models \
    --use_gpu True \
    --is_parallel True

