model_name=roberta_skep_large_en
model_targz=${model_name}.tar.gz
rm -rf $model_name
wget --no-check-certificate https://senta.bj.bcebos.com/skep/$model_targz .
tar zxvf $model_targz
rm -f $model_targz
