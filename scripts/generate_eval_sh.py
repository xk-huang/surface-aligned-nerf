import os.path as osp
from glob import glob

def find_largest_epoch(weights_dir):
    files = glob(osp.join(weights_dir, '*.pth'))
    files = [f for f in files if 'latest' not in f]
    print(f"# Found {len(files)} weights in {weights_dir}: {files}")
    if len(files) == 0:
        raise ValueError("No weights found in {}".format(weights_dir))
    return max([int(osp.basename(f).split('.')[0].split('_')[-1]) for f in files])

h36m_suffix=""
zju_mocap_suffix=""
base_dir = 'data/trained_model/sa-nerf/'
time_ls = [5*60, 10*60, 60*60*2]
gpus_ls = list(range(1,8))

num_gpus = len(gpus_ls)
for _time in time_ls:
    for idx, job_id in enumerate('313 315 377 386 387 390 392 393 394 11 1 5 6 7 8 9'.split(' ')):
        if idx != 0 and idx%num_gpus==0:
            print('wait')
        gpu_id = gpus_ls[idx%num_gpus]
        if int(job_id) > 300:
            weights_dir = f"{base_dir}/zju_mocap/{_time}/test_{job_id}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={job_id}; python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_${{id}}.yaml exp_name zju_mocap/{_time}/test_${{id}} gpus [{gpu_id}] test.epoch {weight_epoch} result_dir data/result/novel_view num_train_frame 300 num_novel_pose_frame 300 training_view "0,1,3,4,5,6,7,9,10,11,12,13,15,16,17,18,19,21,22" &')
        else:
            weights_dir = f"{base_dir}/h36m/{_time}/test_{job_id}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={job_id}; python run.py --type evaluate --cfg_file configs/h36m_exp/latent_xyzc_s${{id}}p.yaml exp_name h36m/{_time}/test_${{id}} gpus [{gpu_id}] test.epoch {weight_epoch} result_dir data/result/novel_view num_train_frame 300 num_novel_pose_frame 300 &')
    print('wait')

for _time in time_ls:
    for idx, job_id in enumerate('313 315 377 386 387 390 392 393 394 11 1 5 6 7 8 9'.split(' ')):
        if idx != 0 and idx%num_gpus==0:
            print('wait')
        gpu_id = gpus_ls[idx%num_gpus]
        if int(job_id) > 300:
            weights_dir = f"{base_dir}/zju_mocap/{_time}/test_{job_id}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={job_id}; python run.py --type evaluate --cfg_file configs/zju_mocap_exp/multi_view_${{id}}.yaml exp_name zju_mocap/{_time}/test_${{id}} gpus [{gpu_id}] test.epoch {weight_epoch} result_dir data/result/novel_pose test_novel_pose True num_train_frame 300 num_novel_pose_frame 300 training_view "0,1,3,4,5,6,7,9,10,11,12,13,15,16,17,18,19,21,22" &')
        else:
            weights_dir = f"{base_dir}/h36m/{_time}/test_{job_id}/"
            weight_epoch = find_largest_epoch(weights_dir)
            print(f'id={job_id}; python run.py --type evaluate --cfg_file configs/h36m_exp/latent_xyzc_s${{id}}p.yaml exp_name h36m/{_time}/test_${{id}} gpus [{gpu_id}] test.epoch {weight_epoch} result_dir data/result/novel_pose test_novel_pose True num_train_frame 300 num_novel_pose_frame 300 &')
    print('wait')