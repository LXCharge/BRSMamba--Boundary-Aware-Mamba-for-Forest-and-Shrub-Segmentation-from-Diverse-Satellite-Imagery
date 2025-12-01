export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=atmbnetv8_no_edge
# MODEL_NAME=atmbnetv8
DATA_SET=vaihingen
# python test_train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_8 -hd 4 -ws 8
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
MODEL_NAME=atmbnetv8
DATA_SET=vaihingen
# python test_train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_8 -hd 4 -ws 8
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
MODEL_NAME=rssformer
python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=transunet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=mifnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
DATA_SET=loveda
MODEL_NAME=atmbnetv8_no_edge
# python test_train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
DATA_SET=loveda
# MODEL_NAME=atmbnetv8
# python test_train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
MODEL_NAME=rssformer
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=transunet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=mifnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb

DATA_SET=potsdam_t
MODEL_NAME=atmbnetv8_no_edge
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
DATA_SET=potsdam_t
MODEL_NAME=atmbnetv8
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME}_temp --rgb
MODEL_NAME=rssformer
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=transunet
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=mifnet
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb

# python test_train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
DATA_SET=gid
MODEL_NAME=atmbnetv8
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=cmtfnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=danet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=dcswin
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=deeplabv3plus
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=manet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=pspnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=rs3mamba
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=segformer
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=sfanet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=unetformer
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=unetmamba
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=rssformer
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=transunet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=mifnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=rsmamba
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb
MODEL_NAME=mifnet
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb

MODEL_NAME=danet
DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=dcswin
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=manet
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=pspnet
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=rs3mamba
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=rsmamba
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=segformer
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=unetformer
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 

# MODEL_NAME=unetmamba
# DATA_SET=vaihingen
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=loveda
# python ${DATA_SET}_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 
# DATA_SET=potsdam_t
# python potsdam_test.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -o ./output/${DATA_SET}/${MODEL_NAME} --rgb 