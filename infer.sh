export CUDA_VISIBLE_DEVICES=4,5
python sentiment_classify.py \
    --test_data_path ./data/test_data/corpus.test \
    --word_dict_path ./data/train.vocab \
    --mode infer \
    --model_path ./models/epoch9/ \
    --use_gpu True 

