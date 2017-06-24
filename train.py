# to run: python3.5 train.py -
# or: nohup python3.5 train.py &

import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box
from inspect import getsourcefile

from estimator import Model
from worker import Worker

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        img = np.squeeze(img)
        return img

def make_env():
    env_spec = gym.spec('ppaquette/DoomBasic-v0')
    env_spec.id = 'DoomBasic-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=80, height=80, grayscale=True)
    return e

NUM_WORKERS = 1
T_MAX = 5
VALID_ACTIONS = [0, 1, 2, 3]

with tf.device("/cpu:0"):

  with tf.variable_scope("global") as vs:
    model_net = Model(num_outputs=len(VALID_ACTIONS))

  global_counter = itertools.count()

  workers = []
  for worker_id in range(NUM_WORKERS):
    worker = Worker(
      name="worker_{}".format(worker_id),
      env=make_env(),
      model_net=model_net,
      discount_factor = 0.99,
      t_max=T_MAX,
      global_counter=global_counter)
    workers.append(worker)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  worker_threads = []
  for worker in workers:
    worker_fn = lambda: worker.run(sess, coord)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  coord.join(worker_threads)