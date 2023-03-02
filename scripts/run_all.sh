#!/bin/sh

ids=($(ls configs/zju_mocap_exp | grep -E '[0-9]+' -o))
for id in ${ids[@]}; do
    python scripts/run_timer.py -t 15 "python train_net.py --cfg_file configs/zju_mocap_exp/multi_view_${id}.yaml exp_name zju_mocap/test_${id} resume False"
done

ids=($(ls configs/h36m_exp | grep -E '[0-9]+' -o))
for id in ${ids[@]}; do
    python scripts/run_timer.py -t 15 "python train_net.py --cfg_file configs/h36m_exp/latent_xyzc_s${id}p.yaml exp_name h36m/test_${id} resume False"
done