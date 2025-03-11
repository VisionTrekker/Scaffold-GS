voxel_size = 0.001
update_init_factor = 16
appearance_dim = 0
ratio = 1
gpu = 3
iterations = 30_000

import os

# scenes = {
#     'town-train': 'cuda',
#     'town2-train': 'cuda',
#     'building1-train': 'cpu',
#     'building2-train': 'cpu',
#     'building3-train': 'cpu',
# }

# for idx, scene in enumerate(['town-train', 'town2-train', 'building1-train', 'building2-train', 'building3-train']):

scenes = {
    'building1-train': 'cpu',
}

# for idx, scene in enumerate(scenes.items()):
#    print('---------------------------------------------------------------------------------')
#    one_cmd = f'python train.py -s /data2/liuzhi/Dataset/3DGS_Dataset/{scene[0]} -m outputs/{scene[0]} --lod 0 --gpu {gpu} --voxel_size {voxel_size} --update_init_factor {update_init_factor} --appearance_dim {appearance_dim} --ratio {ratio} --iterations {iterations} --port 6102 --resolution 1 --data_device "{scene[1]}"'
#    print(one_cmd)
#    os.system(one_cmd)

for idx, scene in enumerate(['building1-train']):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python render.py -m outputs/{scene}'
    print(one_cmd)
    os.system(one_cmd)

    print('---------------------------------------------------------------------------------')
    one_cmd = f'python metrics.py -m outputs/{scene}'
    print(one_cmd)
    os.system(one_cmd)


# python train.py -s /media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/Dataset/3DGS_Dataset/town-train -m output_gt/town-train --lod 0 --gpu 0 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 30_000 --port 6102 --resolution 1 --data_device "cuda"
