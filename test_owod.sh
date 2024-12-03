#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB, S-OWODB or nu-OWODB

python test.py --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --ckpt ${BENCHMARK}/t1.pth

python test.py --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --ckpt ${BENCHMARK}/t2.pth

python test.py --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --ckpt ${BENCHMARK}/t3.pth

python test.py --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --ckpt ${BENCHMARK}/t4.pth
