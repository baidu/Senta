export CUDA_VISIBLE_DEVICES=0           # available GPU indices
export OPENBLAS_NUM_THREADS=4           # available cpu threads (for openblas setting)
export MKL_NUM_threads=4                # available cpu threads (for mkl setting)
python sentiment_classify.py \
    --model_type bilstm_net \
    --train_data_path ./data/train_data/corpus.train \
    --word_dict_path ./data/train.vocab \
    --mode train \
    --model_path ./models \
    --use_gpu True \
    --is_parallel True

