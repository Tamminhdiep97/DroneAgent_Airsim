import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
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
            "prev_distance": 0,
            "distance": 0
        }

        self.thresh_dist = 80
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
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
            float(self.pts[2]), 10
        ).join()
        self.get_DroneState()
        self.get_RealDistance()
        self.state['prev_distance'] = self.state['distance']
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    def get_DroneState(self):
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)

        im_final = np.array(image.resize((128, 128)).convert("L"))

        return im_final.reshape([128, 128, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        while responses[0].height == 0:
            logger.error('Airsim return [0, 0] image, resend API')
            responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.get_DroneState()
        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            4,
        ).join()

    def get_RealDistance(self):
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )
        dist_actual = np.linalg.norm((quad_pt - self.destination))
        self.state['prev_distance'] = self.state['distance']
        self.state['distance'] = dist_actual
        return dist_actual

    def _compute_reward(self):
        beta = 2
        # sigmoid_factor = 10
        # epsilon = 1e-3
        truncated = False

        if self.state["collision"]:
            reward = -1000
        else:
            # dist = 10000000
            # for i in range(0, len(pts) - 1):
            dist = self.get_RealDistance()
            # dist = min(
            #     dist,
            #     dist_actual
            # )
            logger.info('Distance to destination: {}'.format(dist))
            if dist > self.thresh_dist:
                reward = -500
                truncated = True
                return reward, 0, truncated
            else:
                # reward_dist = math.exp(-beta * dist) - 0.5
                # reward_speed = (
                #     np.linalg.norm(
                #         [
                #             self.state["velocity"].x_val,
                #             self.state["velocity"].y_val,
                #             self.state["velocity"].z_val,
                #         ]
                #     )
                #     - 0.5
                # )
                # add *2 to make sure distance is more important than speed
                # reward = reward_dist*2 + reward_speed
                # test reward only on distance to destination
                # reward = beta * (self.state['prev_distance'] - dist) - dist
                distance_improvement = self.state['prev_distance'] - dist
                distance_base_reward = (10 * (1 - dist/self.thresh_dist) + distance_improvement)
                logger.info('Distance improvement: {}'.format(distance_improvement))
                reward = beta * np.sinh(0.01*distance_base_reward)
                # reward = reward_dist

        done = False
        if reward <= -100:
            done = True
        else:
            # high reward if drone position near destination position
            if dist <= 10:
                reward += 100
                done = True

        return reward, done, truncated

    def get_obs(self):
        self.obs = dict(
            {
                'image': self._get_obs(),
                'destination': np.array(self.destination).astype(np.float32),
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
        reward, terminated, truncated = self._compute_reward()
        logger.info(
            'Previous Distance: {}, Distance: {}'.format(
                self.state['prev_distance'],
                self.state['distance']
            )
        )
        logger.info(
            'Reward: {}, Terminated: {}, Truncated: {}'.format(
                reward, terminated, truncated
            )
        )

        return self.obs, reward, terminated, truncated, {}
        # return {'image': obs, reward, done, self.state

    def reset(self, seed=None, **kwargs):
        logger.info('Reset')
        logger.info('*-'*40)
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
