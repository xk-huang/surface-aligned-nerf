for _time in [5*60, 10*60, 60*60*2]:
    num_gpus = 8
    for i, j in enumerate('313 315 377 386 387 390 392 393 394 11 1 5 6 7 8 9'.split(' ')):
        if i != 0 and i%num_gpus==0:
            print('wait')
        if int(j) > 300:
            print(f'id={j}; python scripts/run_timer.py -t {_time} "python train_net.py --cfg_file configs/zju_mocap_exp/multi_view_${{id}}.yaml exp_name zju_mocap/{_time}/test_${{id}} resume False gpus [{i%num_gpus}]" &')
        else:
            print(f'id={j}; python scripts/run_timer.py -t {_time} "python train_net.py --cfg_file configs/h36m_exp/latent_xyzc_s${{id}}p.yaml exp_name h36m/{_time}/test_${{id}} resume False gpus [{i%num_gpus}]" &')
    print('wait')