import tensorflow as tf

from functools import reduce
from helpers import static_size


def load_network(
        datasource, arch, num_classes,
        initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
        ):
    networks = {
        'lenet300': lambda: LeNet300(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            ),
        'lenet5': lambda: LeNet5(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap),
        'alexnet-v1': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=1),
        'alexnet-v2': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=2),
        'vgg-c': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='C'),
        'vgg-d': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='D'),
        'vgg-like': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='like'),
        'resnet': lambda: ResNet20_V1(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes)
    }
    return networks[arch]()


def get_initializer(initializer, dtype):
    if initializer == 'zeros':
        return tf.zeros_initializer()
    elif initializer == 'vs':
        return tf.compat.v1.variance_scaling_initializer(dtype=dtype)
    else:
        raise NotImplementedError


class LeNet300(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet300'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = {k: tf.Variable(self.weights_bp[k].initialized_value(), trainable=True, name='ap/'+k) for k in self.weights_bp}
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.compat.v1.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.compat.v1.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.compat.v1.variable_scope(scope):
            weights['w1'] = tf.compat.v1.get_variable('w1', [784, 300], **w_params)
            weights['w2'] = tf.compat.v1.get_variable('w2', [300, 100], **w_params)
            weights['w3'] = tf.compat.v1.get_variable('w3', [100, 10], **w_params)
            weights['b1'] = tf.compat.v1.get_variable('b1', [300], **b_params)
            weights['b2'] = tf.compat.v1.get_variable('b2', [100], **b_params)
            weights['b3'] = tf.compat.v1.get_variable('b3', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs_flat = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        fc1 = tf.matmul(inputs_flat, weights['w1']) + weights['b1']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w2']) + weights['b2']
        fc2 = tf.nn.relu(fc2)
        fc3 = tf.matmul(fc2, weights['w3']) + weights['b3']
        return fc3


class LeNet5(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet5'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = {k: tf.Variable(self.weights_bp[k].initialized_value(), trainable=True, name='ap/'+k) for k in self.weights_bp}
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.compat.v1.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.compat.v1.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.compat.v1.variable_scope(scope):
            weights['w1'] = tf.compat.v1.get_variable('w1', [5, 5, 1, 20], **w_params)
            weights['w2'] = tf.compat.v1.get_variable('w2', [5, 5, 20, 50], **w_params)
            weights['w3'] = tf.compat.v1.get_variable('w3', [800, 500], **w_params)
            weights['w4'] = tf.compat.v1.get_variable('w4', [500, 10], **w_params)
            weights['b1'] = tf.compat.v1.get_variable('b1', [20], **b_params)
            weights['b2'] = tf.compat.v1.get_variable('b2', [50], **b_params)
            weights['b3'] = tf.compat.v1.get_variable('b3', [500], **b_params)
            weights['b4'] = tf.compat.v1.get_variable('b4', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        conv1 = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'VALID') + weights['b1']
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        conv2 = tf.nn.conv2d(pool1, weights['w2'], [1, 1, 1, 1], 'VALID') + weights['b2']
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        flatten = tf.reshape(pool2, [-1, reduce(lambda x, y: x*y, pool2.shape.as_list()[1:])])
        fc1 = tf.matmul(flatten, weights['w3']) + weights['b3']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w4']) + weights['b4'] # logits
        return fc2


class AlexNet(object):
    ''' Similar to Alexnet in terms of the total number of conv and fc layers.

    Conv layers:
        The size of kernels and the number of conv filters are the same as the original.
        Due to the smaller input size (CIFAR rather than IMAGENET) we use different strides.
    FC layers:
        The size of fc layers are controlled by k (multiplied by 1024).
        In the original Alexnet, k=4 making the size of largest fc layers to be 4096.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 k,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.k = k
        self.name = 'alexnet'
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = {k: tf.Variable(self.weights_bp[k].initialized_value(), trainable=True, name='ap/'+k) for k in self.weights_bp}
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.compat.v1.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.compat.v1.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        k = self.k
        weights = {}
        with tf.compat.v1.variable_scope(scope):
            weights['w1'] = tf.compat.v1.get_variable('w1', [11, 11, 3, 96], **w_params)
            weights['w2'] = tf.compat.v1.get_variable('w2', [5, 5, 96, 256], **w_params)
            weights['w3'] = tf.compat.v1.get_variable('w3', [3, 3, 256, 384], **w_params)
            weights['w4'] = tf.compat.v1.get_variable('w4', [3, 3, 384, 384], **w_params)
            weights['w5'] = tf.compat.v1.get_variable('w5', [3, 3, 384, 256], **w_params)
            weights['w6'] = tf.compat.v1.get_variable('w6', [256, 1024*k], **w_params)
            weights['w7'] = tf.compat.v1.get_variable('w7', [1024*k, 1024*k], **w_params)
            weights['w8'] = tf.compat.v1.get_variable('w8', [1024*k, self.num_classes], **w_params)
            weights['b1'] = tf.compat.v1.get_variable('b1', [96], **b_params)
            weights['b2'] = tf.compat.v1.get_variable('b2', [256], **b_params)
            weights['b3'] = tf.compat.v1.get_variable('b3', [384], **b_params)
            weights['b4'] = tf.compat.v1.get_variable('b4', [384], **b_params)
            weights['b5'] = tf.compat.v1.get_variable('b5', [256], **b_params)
            weights['b6'] = tf.compat.v1.get_variable('b6', [1024*k], **b_params)
            weights['b7'] = tf.compat.v1.get_variable('b7', [1024*k], **b_params)
            weights['b8'] = tf.compat.v1.get_variable('b8', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 4 if self.datasource == 'tiny-imagenet' else 2
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1,init_st,init_st,1], 'SAME') + weights['b1']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w2'], [1, 2, 2, 1], 'SAME') + weights['b2']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w3'], [1, 2, 2, 1], 'SAME') + weights['b3']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w4'], [1, 2, 2, 1], 'SAME') + weights['b4']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w5'], [1, 2, 2, 1], 'SAME') + weights['b5']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w6']) + weights['b6']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w7']) + weights['b7']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w8']) + weights['b8'] # logits
        return inputs


class VGG(object):
    '''
    Similar to the original VGG.
    Available models:
        - VGG-C
        - VGG-D
        - VGG-like

    Differences:
        The number of parameters in conv layers are the same as the original.
        The number of parameters in fc layers are reduced to 512 (4096 -> 512).
        The number of total parameters are different, not just because of the size of fc layers,
        but also due to the fact that the first fc layer receives 1x1 image rather than 7x7 image
        because the input is CIFAR not IMAGENET.
        No dropout is used. Instead, batch norm is used.

    Other refereneces.
        (1) The original paper:
        - paper: https://arxiv.org/pdf/1409.1556.pdf
        - code: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
        * Dropout between fc layers.
        * There is no BatchNorm.
        (2) VGG-like by Zagoruyko, adapted for CIFAR-10.
        - project and code: http://torch.ch/blog/2015/07/30/cifar.html
        * Differences to the original VGG-16 (1):
            - # of fc layers 3 -> 2, so there are 15 (learnable) layers in total.
            - size of fc layers 4096 -> 512.
            - use BatchNorm and add more Dropout.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 version,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.version = version
        self.name = 'VGG-{}'.format(version)
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = {k: tf.Variable(self.weights_bp[k].initialized_value(), trainable=True, name='ap/'+k) for k in self.weights_bp}
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.compat.v1.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.compat.v1.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.compat.v1.variable_scope(scope):
            weights['w1'] = tf.compat.v1.get_variable('w1', [3, 3, 3, 64], **w_params)
            weights['w2'] = tf.compat.v1.get_variable('w2', [3, 3, 64, 64], **w_params)
            weights['w3'] = tf.compat.v1.get_variable('w3', [3, 3, 64, 128], **w_params)
            weights['w4'] = tf.compat.v1.get_variable('w4', [3, 3, 128, 128], **w_params)
            weights['b1'] = tf.compat.v1.get_variable('b1', [64], **b_params)
            weights['b2'] = tf.compat.v1.get_variable('b2', [64], **b_params)
            weights['b3'] = tf.compat.v1.get_variable('b3', [128], **b_params)
            weights['b4'] = tf.compat.v1.get_variable('b4', [128], **b_params)
            if self.version == 'C':
                weights['w5'] = tf.compat.v1.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.compat.v1.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.compat.v1.get_variable('w7', [1, 1, 256, 256], **w_params)
                weights['w8'] = tf.compat.v1.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.compat.v1.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.compat.v1.get_variable('w10', [1, 1, 512, 512], **w_params)
                weights['w11'] = tf.compat.v1.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.compat.v1.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.compat.v1.get_variable('w13', [1, 1, 512, 512], **w_params)
                weights['b5'] = tf.compat.v1.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.compat.v1.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.compat.v1.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.compat.v1.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.compat.v1.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.compat.v1.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.compat.v1.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.compat.v1.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.compat.v1.get_variable('b13', [512], **b_params)
            elif self.version == 'D' or self.version == 'like':
                weights['w5'] = tf.compat.v1.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.compat.v1.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.compat.v1.get_variable('w7', [3, 3, 256, 256], **w_params)
                weights['w8'] = tf.compat.v1.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.compat.v1.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.compat.v1.get_variable('w10', [3, 3, 512, 512], **w_params)
                weights['w11'] = tf.compat.v1.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.compat.v1.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.compat.v1.get_variable('w13', [3, 3, 512, 512], **w_params)
                weights['b5'] = tf.compat.v1.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.compat.v1.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.compat.v1.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.compat.v1.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.compat.v1.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.compat.v1.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.compat.v1.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.compat.v1.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.compat.v1.get_variable('b13', [512], **b_params)
            weights['w14'] = tf.compat.v1.get_variable('w14', [512, 512], **w_params)
            weights['b14'] = tf.compat.v1.get_variable('b14', [512], **b_params)
            if not self.version == 'like':
                weights['w15'] = tf.compat.v1.get_variable('w15', [512, 512], **w_params)
                weights['w16'] = tf.compat.v1.get_variable('w16', [512, self.num_classes], **w_params)
                weights['b15'] = tf.compat.v1.get_variable('b15', [512], **b_params)
                weights['b16'] = tf.compat.v1.get_variable('b16', [self.num_classes], **b_params)
            else:
                weights['w15'] = tf.compat.v1.get_variable('w15', [512, self.num_classes], **w_params)
                weights['b15'] = tf.compat.v1.get_variable('b15', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _conv_block(inputs, bn_params, filt, st=1):
            inputs = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
            inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            return inputs

        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1

        inputs = _conv_block(inputs, bn_params, {'w': weights['w1'], 'b': weights['b1']}, init_st)
        inputs = _conv_block(inputs, bn_params, {'w': weights['w2'], 'b': weights['b2']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w3'], 'b': weights['b3']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w4'], 'b': weights['b4']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w5'], 'b': weights['b5']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w6'], 'b': weights['b6']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w7'], 'b': weights['b7']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w8'], 'b': weights['b8']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w9'], 'b': weights['b9']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w10'], 'b': weights['b10']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w11'], 'b': weights['b11']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w12'], 'b': weights['b12']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w13'], 'b': weights['b13']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        assert reduce(lambda x, y: x*y, inputs.shape.as_list()[1:3]) == 1

        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w14']) + weights['b14']
        inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        if not self.version == 'like':
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']
            inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.matmul(inputs, weights['w16']) + weights['b16']
        else:
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']

        return inputs


class ResNet20_V1(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.name = 'ResNet20-V1'
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = {k: tf.Variable(self.weights_bp[k].initialized_value(), trainable=True, name='ap/'+k) for k in self.weights_bp}
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.compat.v1.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.compat.v1.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.compat.v1.variable_scope(scope):
            weights['w1'] = tf.compat.v1.get_variable('w1', [3, 3, 3, 16], **w_params)
            weights['w2'] = tf.compat.v1.get_variable('w2', [3, 3, 16, 16], **w_params)
            weights['w3'] = tf.compat.v1.get_variable('w3', [3, 3, 16, 16], **w_params)
            weights['w4'] = tf.compat.v1.get_variable('w4', [3, 3, 16, 16], **w_params)
            weights['w5'] = tf.compat.v1.get_variable('w5', [3, 3, 16, 16], **w_params)
            weights['w6'] = tf.compat.v1.get_variable('w6', [3, 3, 16, 16], **w_params)
            weights['w7'] = tf.compat.v1.get_variable('w7', [3, 3, 16, 16], **w_params)
            weights['wsp1'] = tf.compat.v1.get_variable('wsp1', [1, 1, 16, 32], **w_params)
            weights['w8'] = tf.compat.v1.get_variable('w8', [3, 3, 16, 32], **w_params)
            weights['w9'] = tf.compat.v1.get_variable('w9', [3, 3, 32, 32], **w_params)
            weights['w10'] = tf.compat.v1.get_variable('w10', [3, 3, 32, 32], **w_params)
            weights['w11'] = tf.compat.v1.get_variable('w11', [3, 3, 32, 32], **w_params)
            weights['w12'] = tf.compat.v1.get_variable('w12', [3, 3, 32, 32], **w_params)
            weights['w13'] = tf.compat.v1.get_variable('w13', [3, 3, 32, 32], **w_params)
            weights['wsp2'] = tf.compat.v1.get_variable('wsp2', [1, 1, 32, 64], **w_params)
            weights['w14'] = tf.compat.v1.get_variable('w14', [3, 3, 32, 64], **w_params)
            weights['w15'] = tf.compat.v1.get_variable('w15', [3, 3, 64, 64], **w_params)
            weights['w16'] = tf.compat.v1.get_variable('w16', [3, 3, 64, 64], **w_params)
            weights['w17'] = tf.compat.v1.get_variable('w17', [3, 3, 64, 64], **w_params)
            weights['w18'] = tf.compat.v1.get_variable('w18', [3, 3, 64, 64], **w_params)
            weights['w19'] = tf.compat.v1.get_variable('w19', [3, 3, 64, 64], **w_params)
            weights['wfc'] = tf.compat.v1.get_variable('wfc', [1024, self.num_classes], **w_params)
            weights['b1'] = tf.compat.v1.get_variable('b1', [16], **b_params)
            weights['b2'] = tf.compat.v1.get_variable('b2', [16], **b_params)
            weights['b3'] = tf.compat.v1.get_variable('b3', [16], **b_params)
            weights['b4'] = tf.compat.v1.get_variable('b4', [16], **b_params)
            weights['b5'] = tf.compat.v1.get_variable('b5', [16], **b_params)
            weights['b6'] = tf.compat.v1.get_variable('b6', [16], **b_params)
            weights['b7'] = tf.compat.v1.get_variable('b7', [16], **b_params)
            weights['bsp1'] = tf.compat.v1.get_variable('bsp1', [32], **b_params)
            weights['b8'] = tf.compat.v1.get_variable('b8', [32], **b_params)
            weights['b9'] = tf.compat.v1.get_variable('b9', [32], **b_params)
            weights['b10'] = tf.compat.v1.get_variable('b10', [32], **b_params)
            weights['b11'] = tf.compat.v1.get_variable('b11', [32], **b_params)
            weights['b12'] = tf.compat.v1.get_variable('b12', [32], **b_params)
            weights['b13'] = tf.compat.v1.get_variable('b13', [32], **b_params)
            weights['bsp2'] = tf.compat.v1.get_variable('bsp2', [64], **b_params)
            weights['b14'] = tf.compat.v1.get_variable('b14', [64], **b_params)
            weights['b15'] = tf.compat.v1.get_variable('b15', [64], **b_params)
            weights['b16'] = tf.compat.v1.get_variable('b16', [64], **b_params)
            weights['b17'] = tf.compat.v1.get_variable('b17', [64], **b_params)
            weights['b18'] = tf.compat.v1.get_variable('b18', [64], **b_params)
            weights['b19'] = tf.compat.v1.get_variable('b19', [64], **b_params)
            weights['bfc'] = tf.compat.v1.get_variable('bfc', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _conv_block(inputs, bn_params, filt, st1=1, st2=1, subsampling=False, subsampling_filt={}):
            padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
            shortcut = inputs
            inputs = tf.nn.conv2d(inputs, filt['c1'], [1, st1, st1, 1], padding) + filt['b1']
            inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, filt['c2'], [1, st2, st2, 1], padding) + filt['b2']
            inputs = tf.compat.v1.layers.batch_normalization(inputs, **bn_params)
            if subsampling:
                shortcut = tf.nn.conv2d(shortcut, subsampling_filt['c'], [1, 2, 2, 1], 'VALID') + subsampling_filt['b']
            inputs = inputs + shortcut
            inputs = tf.nn.relu(inputs)
            return inputs

        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }

        # 3 * 3 convolution layer
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'SAME') + weights['b1']
        # Layer 1
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w2'], 'b1': weights['b2'], 'c2': weights['w3'], 'b2': weights['b3']},
            st1=1, st2=1)
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w4'], 'b1': weights['b4'], 'c2': weights['w5'], 'b2': weights['b5']},
            st1=1, st2=1)
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w6'], 'b1': weights['b6'], 'c2': weights['w7'], 'b2': weights['b7']},
            st1=1, st2=1)
        # Layer 2
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w8'], 'b1': weights['b8'], 'c2': weights['w9'], 'b2': weights['b9']},
            st1=2, st2=1, subsampling=True, subsampling_filt={'c': weights['wsp1'], 'b': weights['bsp1']})
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w10'], 'b1': weights['b10'], 'c2': weights['w11'], 'b2': weights['b11']},
            st1=1, st2=1)
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w12'], 'b1': weights['b12'], 'c2': weights['w13'], 'b2': weights['b13']},
            st1=1, st2=1)
        # Layer 3
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w14'], 'b1': weights['b14'], 'c2': weights['w15'], 'b2': weights['b15']},
            st1=2, st2=1, subsampling=True, subsampling_filt={'c': weights['wsp2'], 'b': weights['bsp2']})
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w16'], 'b1': weights['b16'], 'c2': weights['w17'], 'b2': weights['b17']},
            st1=1, st2=1)
        inputs = _conv_block(inputs, bn_params, 
            {'c1': weights['w18'], 'b1': weights['b18'], 'c2': weights['w19'], 'b2': weights['b19']},
            st1=1, st2=1)
        # Average pooling + fully connected layer
        inputs = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['wfc']) + weights['bfc'] # logits
        return inputs
