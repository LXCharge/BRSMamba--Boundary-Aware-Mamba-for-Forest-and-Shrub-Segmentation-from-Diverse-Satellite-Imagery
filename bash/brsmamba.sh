export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=brsmamba

DATA_SET=vaihingen
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
DATA_SET=potsdam_t
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
DATA_SET=loveda
python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py

