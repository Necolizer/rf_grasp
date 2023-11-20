from pyrfuniverse.envs.gym_wrapper_env import RFUniverseGymWrapper
import pyrfuniverse.attributes as attr
import cv2
import numpy as np
from gym import spaces
from gym.utils import seeding
import copy
import os
import random


class ElfinAG145Env(RFUniverseGymWrapper):
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            max_episode_length,
            reward_type,
            tolerance,
            load_object = True,
            seed = 1234,
            executable_file = None,
            scene_file = None,
            asset_bundle_file = None,
            assets: list = [],
            bin = 64,
            movable_joints = 6,
            raw_img_shape = [1280, 720],
            use_depth_img = False,
            resized_img_shape = [224, 224],
    ):
        super().__init__(
            executable_file,
            scene_file,
            assets=assets
        )
        self.max_steps = max_episode_length
        self.reward_type = reward_type
        self.tolerance = tolerance
        self.load_object = load_object
        self.asset_bundle_file = asset_bundle_file

        # self.bin = bin
        self.movable_joints = movable_joints
        self.raw_img_width = raw_img_shape[0]
        self.raw_img_height = raw_img_shape[1]
        self.use_depth_img = use_depth_img
        self.resized_img_shape = (resized_img_shape[0], resized_img_shape[1])
        self.init_pos = [-87.91730219774908, -16.081455344020732, 69.81835997930848, -170.05405962349167, -88.43722201810024, 37.37410139329363]

        self.seed(seed)
        self._env_setup()
        
        self.t = 0
        self.low = np.array([-175.0, -135.0, -150.0, -175.0, -147.0, -175.0, -1])
        self.high = np.array([175.0, 135.0, 150.0, 175.0, 147.0, 175.0, 1])
        self.action_space = spaces.Box(
            low=self.low,  high=self.high, dtype=np.float32
        )
        # self.action_space = spaces.MultiDiscrete([self.bin for i in range(self.movable_joints)]+[2])
        obs = self._get_obs()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=np.float32),
            'state': spaces.Box(-np.inf, np.inf, shape=obs['state'].shape, dtype=np.float32)
        })

    def joints_angular_range(self):
        joints = [
            [-175.0,175.0],
            [-135.0,135.0],
            [-150.0,150.0],
            [-175.0,175.0],
            [-147.0,147.0],
            [-175.0,175.0],
        ]
        return np.array(joints, dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Params:
            action: numpy array.
        """
        
        self.robot.SetJointPosition(action[:-1].tolist())

        if action[-1] > 0:
            self.gripper.GripperOpen()
        else:
            self.gripper.GripperClose()

        self._step()
        self.t += 1

        obs = self._get_obs()
        done = False
        
        reward, isSuccess = self.reward_and_success_check()

        info = {
            'is_success': isSuccess
        }

        if self.t == self.max_steps:
            done = True

        return obs, reward, done, info

    def reset(self):
        super().reset()
        self.t = 0

        if self.load_object:
            self._reset_object()

        self.robot.SetJointPositionDirectly(self.init_pos)
        self.gripper.GripperOpen()

        self._step()

        return self._get_obs()

    def seed(self, seed=1234):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._step()

    def reward_and_success_check(self):
        # 1. Distance-based

        achieved_goal = np.array(self.gripper.data['positions'][-1])

        # averaged origins
        # desired_goal = np.mean(np.array(self.akb.data['positions']), axis=0) 
        # center of mass
        mass = np.array(self.akb.data['mass'])
        desired_goal = (mass/np.sum(mass)) @ self.akb.data['center_of_mass']

        distance = self._compute_goal_distance(achieved_goal, desired_goal)

        if self.reward_type == 'sparse':
            distance_reward = -(distance > self.tolerance).astype(np.float32)
        else:
            distance_reward = -distance

        distance_isSuccess = (distance < self.tolerance).astype(np.float32)

        # 2. Collision-based

        # 3. Physics-based

        # reward, isSuccess
        return distance_reward, distance_isSuccess

    def _env_setup(self):

        self.robot =  self.attrs[55751]
        self.robot.SetJointPositionDirectly(self.init_pos)
        self.gripper = self.attrs[557510]
        self.gripper.GripperOpen()

        self.camera = self.InstanceObject(name='Camera', id=123456, attr_type=attr.CameraAttr)

        # FIXME camera pos and ori random
        self.camera.SetTransform(position=[1.5, 0.8, 0], rotation=[0, 0, 0])
        # self.camera.LookAt(target=[i+j for i, j in zip(self.robot.data['position'], self.gripper.data['position'])])
        self.camera.LookAt(target=self.robot.data['position'])

        self.akb = None
        if self.load_object:
            self._reset_object()

        self._step()

    def _get_obs(self):
        self.camera.GetRGB(width=self.raw_img_width, height=self.raw_img_height)
        if self.use_depth_img:
            self.camera.GetDepth(width=self.raw_img_width, height=self.raw_img_height, zero_dis=0.1, one_dis=5)
            # self.camera.GetDepthEXR(width=self.raw_img_width, height=self.raw_img_height)
        
        self._step()

        rgb = np.frombuffer(self.camera.data['rgb'], dtype=np.uint8)
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, self.resized_img_shape)
        
        if self.use_depth_img:
            depth = np.frombuffer(self.camera.data['depth'], dtype=np.uint8)
            depth = cv2.imdecode(depth, cv2.IMREAD_GRAYSCALE)
            depth = cv2.bitwise_not(cv2.resize(depth, self.resized_img_shape))
            obs = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)
        else:
            obs = rgb

        # FIXME add camera pos and ori

        return {
            'observation': obs.copy(),
            'state': np.array(self.robot.data['joint_positions']).copy(),
        }

    def _generate_random_float(self, min: float, max: float) -> float:
        assert min < max, \
            'Min value is {}, while max value is {}.'.format(min, max)
        random_float = np.random.rand()
        random_float = random_float * (max - min) + min

        return random_float

    def _compute_goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _reset_object(self):
        if self.akb:
            self.akb.Destroy()

        # FIXME akb loading and randomizing
        self.akb = self.LoadURDF(path=os.path.abspath('/media/amax/NECOStorage/AKB_48_Dataset_v1.0/v1.0/drink/0b6681c4-0e49-11ed-a4e9-ec2e98c7e246/motion_unity_update.urdf'), native_ik=False)
        # FIXME akb randomize pos and ori
        self.akb.SetTransform(position=[random.uniform(0.4, 0.6), 0.03, random.uniform(-0.6, 0.6)], rotation=[0, 0, 0])
        self.akb.SetImmovable(False)
        self.akb.SetAllGravity(True)
        self.akb.GenerateMeshCollider()
        self._step()

    # def debug(self):
    #     a = self.gripper.data['gripper_is_open']
    #     b = self.gripper.data['gripper_is_holding']
    #     print(a)
    #     print(b)

    #     self.gripper.GripperClose()

    #     self._step()

    #     a = self.gripper.data['gripper_is_open']
    #     b = self.gripper.data['gripper_is_holding']
    #     print(a)
    #     print(b)