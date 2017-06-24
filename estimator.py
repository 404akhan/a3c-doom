import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self._build_model() 

    def _build_model(self):
        self.states = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='X')
        self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pi")
        self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_v") 
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.states)[0]
        conv1 = tf.contrib.layers.conv2d(
            self.states, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv1")
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv2 = tf.contrib.layers.conv2d(
            pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv2")
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv3 = tf.contrib.layers.conv2d(
            pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv3")
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
        flattened = tf.contrib.layers.flatten(pool3)

        ### Policy
        fc1_pi = tf.contrib.layers.fully_connected(flattened, 64)
        self.logits_pi = tf.contrib.layers.fully_connected(fc1_pi, self.num_outputs, activation_fn=None)
        self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

        self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")

        gather_indices_pi = tf.range(batch_size) * tf.shape(self.probs_pi)[1] + self.actions
        self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

        self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.01 * self.entropy_pi)
        self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

        ### Value
        fc1_v = tf.contrib.layers.fully_connected(flattened, 128)
        self.logits_v = tf.contrib.layers.fully_connected(inputs=fc1_v, num_outputs=1, activation_fn=None)
        self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

        self.losses_v = tf.squared_difference(self.logits_v, self.targets_v)
        self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")

        # Combine loss
        self.loss = self.loss_pi + self.loss_v
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())
