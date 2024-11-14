#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB, S-OWODB or nu-OWODB

python base_eval.py --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml

#python base_eval.py --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml

#python base_eval.py --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml

#python base_eval.py --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml
