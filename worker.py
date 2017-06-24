import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
from estimator import Model

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

def make_train_op(local_estimator, global_estimator):
  local_grads, _ = zip(*local_estimator.grads_and_vars)
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
  _, global_vars = zip(*global_estimator.grads_and_vars)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())


class Worker(object):
  def __init__(self, name, env, model_net, discount_factor, t_max, global_counter):
    self.name = name
    self.env = env
    self.global_model_net = model_net
    self.discount_factor = discount_factor
    self.t_max = t_max
    self.local_counter = itertools.count()
    self.global_counter = global_counter

    with tf.variable_scope(name):
      self.model_net = Model(model_net.num_outputs)

    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.mnet_train_op = make_train_op(self.model_net, self.global_model_net)

    self._reset()

  def _reset(self):
    self.total_reward = 0
    self.episode_length = 0
    self.action_counter = []

    state = self.env.reset() 
    self.state = np.stack([state] * 4, axis=2)

  def run(self, sess, coord):
    with sess.as_default(), sess.graph.as_default():
      try:
        while not coord.should_stop():
          sess.run(self.copy_params_op)

          transitions = self.run_n_steps(sess)
          self.update(transitions, sess)

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess):
      feed_dict = { self.model_net.states: [state] }
      probs = sess.run(self.model_net.probs_pi, feed_dict)
      return probs[0] 

  def _value_net_predict(self, state, sess):
      feed_dict = { self.model_net.states: [state] }
      logits_v = sess.run(self.model_net.logits_v, feed_dict)
      return logits_v[0]

  def run_n_steps(self, sess):
      transitions = []
      for _ in range(self.t_max):
          action_probs = self._policy_net_predict(self.state, sess)
          action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

          next_state, reward, done, _ = self.env.step(action)
          reward /= 100.
          next_state = np.append(self.state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
         
          self.total_reward += reward
          self.action_counter.append(action)

          transitions.append(Transition(
              state=self.state, action=action, reward=reward, next_state=next_state, done=done))

          if done:
              local_t = next(self.local_counter)
              global_t = next(self.global_counter)
              print("agent {}, global_t {}, local_t {}, total_reward {}, episode_length {}, action distr {}".format(
                  self.name, global_t, local_t, int(self.total_reward*100), self.episode_length, np.bincount(self.action_counter)))
              
              self._reset()
              break
          else:
              self.state = next_state
      return transitions      

  def update(self, transitions, sess):
      reward = 0.0
      if not transitions[-1].done:
        reward = self._value_net_predict(transitions[-1].next_state, sess)

      states = []
      actions = []
      policy_targets = []
      value_targets = []

      for transition in transitions[::-1]:
        reward = transition.reward + self.discount_factor * reward
        policy_target = (reward - self._value_net_predict(transition.state, sess))

        states.append(transition.state)
        actions.append(transition.action)
        policy_targets.append(policy_target)
        value_targets.append(reward)

      feed_dict = {
        self.model_net.states: np.array(states),
        self.model_net.targets_pi: policy_targets,
        self.model_net.targets_v: value_targets,
        self.model_net.actions: actions,
      }

      # Train the global estimators using local gradients
      mnet_loss, _ = sess.run([
        self.model_net.loss,
        self.mnet_train_op
      ], feed_dict)

      return mnet_loss