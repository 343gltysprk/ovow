#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB, S-OWODB or nu-OWODB

python dev.py --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --ckpt yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
python dev.py --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --ckpt ${BENCHMARK}/t1/t1.pth
python dev.py --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --ckpt ${BENCHMARK}/t2/t2.pth
python dev.py --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --ckpt ${BENCHMARK}/t3/t3.pth


