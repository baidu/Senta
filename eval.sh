export CUDA_VISIBLE_DEVICES=0           # available GPU indices
export OPENBLAS_NUM_THREADS=4           # available cpu threads (for openblas setting)
export MKL_NUM_threads=4                # available cpu threads (for mkl setting)
python sentiment_classify.py \
    --test_data_path ./data/test_data/corpus.test \
    --word_dict_path ./data/train.vocab \
    --mode eval \
    --model_path ./models/epoch9/ \
    --use_gpu True

