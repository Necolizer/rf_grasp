import os
import numpy as np
import time
import random
import torch
import cv2

from env.elfin_ag145_env import ElfinAG145Env

env = ElfinAG145Env(
    max_episode_length=160,
    reward_type='dense',
    tolerance=0.02,
    load_object = True,
    seed = 1234,
    executable_file = None,
    scene_file = None,
    asset_bundle_file = None,
    assets=[],
    bin = 64,
    movable_joints = 6,
    raw_img_shape = [512, 512],
    use_depth_img = False,
    resized_img_shape = [224, 224],
)

print('env.action_space.sample() ',env.action_space.sample())
obs,reward,done,info=env.step(env.action_space.sample())
# env.reset()
# obs,reward,done,info=env.step(env.action_space.sample())
print(obs['observation'].shape)
print(obs['state'].shape)
print(obs['state'])
cv2.imshow("rgb", obs['observation'][:,:,:])
cv2.waitKey(0)
env.Pend()


# env.debug()
# env.Pend()