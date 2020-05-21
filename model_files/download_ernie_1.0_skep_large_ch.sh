model_name=ernie_1.0_skep_large_ch
model_targz=${model_name}.tar.gz
rm -rf $model_name
wget --no-check-certificate https://senta.bj.bcebos.com/skep/$model_targz .
tar zxvf $model_targz
rm -f $model_targz
