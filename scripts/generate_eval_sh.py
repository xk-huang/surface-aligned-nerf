import os.path as osp
from glob import glob

def find_largest_epoch(weights_dir):
    files = glob(osp.join(weights_dir, '*.pth'))
    files = [f for f in files if 'latest' not in f]
    print(f"# Found {len(files)} weights in {weights_dir}: {files}")
    if len(files) == 0:
        raise ValueError("No weights found in {}".format(weights_dir))
    return max([int(osp.basename(f).split('.')[0].split('_')[-1]) for f in files])

base_dir = 'data/trained_model/sa-nerf/'
for _time in [5*60, 10*60, 60*60*2]:
    num_gpus = 8
    for i, j in enumerate('313 315 377 386 387 390 392 393 394 11 1 5 6 7 8 9'.split(' ')):
        if i != 0 and i%num_gpus==0:
            print('wait')
        if int(j) > 300:
            weights_dir = f"{base_dir}/zju_mocap/{_time}/test_{j}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={j}; python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_${{id}}.yaml exp_name zju_mocap/{_time}/test_${{id}} gpus [{i%num_gpus}] test.epoch {weight_epoch} cfg.result_dir data/result/novel_view &')
        else:
            weights_dir = f"{base_dir}/h36m/{_time}/test_{j}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={j}; python run.py --type evaluate --cfg_file configs/h36m_exp/latent_xyzc_s${{id}}p.yaml exp_name h36m/{_time}/test_${{id}} gpus [{i%num_gpus}] test.epoch {weight_epoch} cfg.result_dir data/result/novel_view &')
    print('wait')

for _time in [5*60, 10*60, 60*60*2]:
    num_gpus = 8
    for i, j in enumerate('313 315 377 386 387 390 392 393 394 11 1 5 6 7 8 9'.split(' ')):
        if i != 0 and i%num_gpus==0:
            print('wait')
        if int(j) > 300:
            weights_dir = f"{base_dir}/zju_mocap/{_time}/test_{j}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={j}; python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_${{id}}.yaml exp_name zju_mocap/{_time}/test_${{id}} gpus [{i%num_gpus}] test.epoch {weight_epoch} cfg.result_dir data/result/novel_view test_novel_pose True &')
        else:
            weights_dir = f"{base_dir}/h36m/{_time}/test_{j}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={j}; python run.py --type evaluate --cfg_file configs/h36m_exp/latent_xyzc_s${{id}}p.yaml exp_name h36m/{_time}/test_${{id}} gpus [{i%num_gpus}] test.epoch {weight_epoch} cfg.result_dir data/result/novel_view test_novel_pose True &')
    print('wait')