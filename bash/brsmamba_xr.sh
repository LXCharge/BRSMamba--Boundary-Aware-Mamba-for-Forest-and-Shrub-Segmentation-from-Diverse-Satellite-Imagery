export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=atmbnetv8_no_erhindt
DATA_SET=vaihingen
python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=potsdam_t
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=loveda
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
MODEL_NAME=atmbnetv8_no_ssdindt
DATA_SET=vaihingen
python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=potsdam_t
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=loveda
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
MODEL_NAME=atmbnetv8_no_log
DATA_SET=vaihingen
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=potsdam_t
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=loveda
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
MODEL_NAME=atmbnetv8_no_conv
DATA_SET=vaihingen
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
MODEL_NAME=atmbnetv8_no_edge
DATA_SET=vaihingen
python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=potsdam_t
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py
# DATA_SET=loveda
# python train_supervision.py -c ./config/${DATA_SET}/${MODEL_NAME}.py

MODEL_NAME=atmbnetv8_xr
DATA_SET=vaihingen
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_16 -hd 4 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_8 -hd 4 -ws 8
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_4 -hd 4 -ws 4
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_2 -hd 4 -ws 2
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_8_ws_4 -hd 8 -ws 4
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_8_ws_8 -hd 8 -ws 8
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_8_ws_16 -hd 8 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_2_ws_4 -hd 2 -ws 4
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_2_ws_8 -hd 2 -ws 8
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_2_ws_16 -hd 2 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_1_ws_4 -hd 1 -ws 4
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_1_ws_8 -hd 1 -ws 8
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_1_ws_16 -hd 1 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_16_ws_4 -hd 16 -ws 4
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_16_ws_8 -hd 16 -ws 8
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_16_ws_16 -hd 16 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_1_ws_1 -hd 1 -ws 1
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_1_ws_16 -hd 1 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_2_ws_1 -hd 2 -ws 1
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_2_ws_16 -hd 2 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_1 -hd 4 -ws 1
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_4_ws_16 -hd 4 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_8_ws_1 -hd 8 -ws 1
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_8_ws_16 -hd 8 -ws 16
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_16_ws_1 -hd 16 -ws 1
# python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_head_16_ws_16 -hd 16 -ws 16
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_F_F_F_T -hd 2 -ws 8 -in_dt_blk 1
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_F_F_T_T -hd 2 -ws 8 -in_dt_blk 2
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_F_T_T_T -hd 2 -ws 8 -in_dt_blk 3
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_T_F_F_F -hd 2 -ws 8 -in_dt_blk -1
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_T_T_F_F -hd 2 -ws 8 -in_dt_blk -2
python train_supervision_main.py -c ./config/${DATA_SET}/${MODEL_NAME}.py -m ${MODEL_NAME}_in_dt_blk_T_T_T_F -hd 2 -ws 8 -in_dt_blk -3





