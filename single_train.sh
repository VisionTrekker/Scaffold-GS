scene='building1-train'
exp_name='building1-train'

voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=2

ulimit -n 1560

./train.sh -d /data2/liuzhi/Dataset/3DGS_Dataset/${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}