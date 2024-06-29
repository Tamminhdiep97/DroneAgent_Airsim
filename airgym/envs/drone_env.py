import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gymnasium as gym
from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv
from loguru import logger

from . import randomize


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.random = randomize.Random()
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )
        # self.pts = np.array([50.5974, 5.0786, -4.32256]),

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        logger.info('Set up Home Position')
        self.pts = self.random.random_position()
        logger.info('Home Position: {}'.format(self.pts))
        logger.info('Set up Destination')
        self.destination = self.random.random_destination(self.pts)
        logger.info('Destination: {}'.format(self.destination))
        # Set home position and velocity
        self.drone.moveToPositionAsync(
            float(self.pts[0]),
            float(self.pts[1]),
            float(self.pts[2]), 15
        ).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    def _compute_reward(self):
        thresh_dist = 80
        beta = 1

        z = -10
        # pts = [
            # np.array([-0.55265, -31.9786, -19.0225]),
            # np.array([48.59735, -63.3286, -60.07256]),
        # pts =    np.array([193.5974, -55.0786, -46.32256]),
            # np.array([369.2474, 35.32137, -62.5725]),
            # np.array([541.3474, 143.6714, -32.07256]),
        # ]

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            reward = -100
        else:
            dist = 10000000
            # for i in range(0, len(pts) - 1):
            dist_actual = np.linalg.norm((quad_pt - self.pts))
            dist = min(
                dist,
                dist_actual
            )
            logger.info('Distance to destination: {}'.format(dist_actual))
            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
                reward = reward_dist + reward_speed

        done = 0
        if reward <= -10:
            done = 1

        return reward, done

    def get_obs(self):
        self.obs = dict(
            {
                'image': self._get_obs(),
                'destination': np.array(self.pts).astype(np.float32),
                'position': np.array(
                    [
                        self.state['position'].x_val,
                        self.state['position'].y_val,
                        self.state['position'].z_val
                    ]
                ).astype(np.float32)
            },
        )

    def step(self, action):
        self._do_action(action)
        self.get_obs()
        reward, terminated = self._compute_reward()
        logger.info('Reward: {}, Terminated: {}'.format(reward, terminated))
        return self.obs, reward, bool(terminated), bool(terminated), {}
        # return {'image': obs, reward, done, self.state

    def reset(self, seed=None, **kwargs):
        logger.info('Reset')
        logger.info('*-'*30)
        self._setup_flight()
        self.get_obs()
        return (self.obs, {})

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
            logger.info('Action 0: {}'.format(quad_offset))
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
            logger.info('Action 1: {}'.format(quad_offset))
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
            logger.info('Action 2: {}'.format(quad_offset))
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
            logger.info('Action 3: {}'.format(quad_offset))
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
            logger.info('Action 4: {}'.format(quad_offset))
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
            logger.info('Action 5: {}'.format(quad_offset))
        else:
            quad_offset = (0, 0, 0)
            logger.info('Action 6: {}'.format(quad_offset))
        return quad_offset
