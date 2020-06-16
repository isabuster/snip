import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model
import prune
import train
import test


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--datasource', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./data', help='location to dataset')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=[], help='augmentations to perform')
    # Model options
    parser.add_argument('--arch', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--target_sparsity', type=float, default=0.9, help='level of sparsity to achieve')
    # Train options
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--train_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--lr_decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[], help='values for piecewise_constant decay')
    # Initialization
    parser.add_argument('--initializer_w_bp', type=str, default='vs', help='initializer for w before pruning')
    parser.add_argument('--initializer_b_bp', type=str, default='zeros', help='initializer for b before pruning')
    parser.add_argument('--initializer_w_ap', type=str, default='vs', help='initializer for w after pruning')
    parser.add_argument('--initializer_b_ap', type=str, default='zeros', help='initializer for b after pruning')
    # Logging, saving, options
    parser.add_argument('--logdir', type=str, default='logs', help='location for summaries and checkpoints')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    args = parser.parse_args()
    # Add more to args
    args.path_summary = os.path.join(args.logdir, 'summary')
    args.path_model = os.path.join(args.logdir, 'model')
    args.path_assess = os.path.join(args.logdir, 'assess')
    return args


def plot_distribution(sess, layers, pruned=False):
    for idx, var in enumerate(layers):
        if pruned == False:
            layer = np.array(sess.run(var)).flatten()
        else:
            layer = var.flatten()[var.flatten() != 0]
        ax = plt.axes()
        ax.set_axisbelow(True)
        plt.hist(layer, bins=30, label="Weights", density=True, edgecolor='white')
        plt.grid(ls='--')
        left, right = plt.xlim()
        kde_xs = np.linspace(left, right)
        kde = st.gaussian_kde(layer)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel('Density')
        plt.xlabel('Weights')
        if pruned == False:
            plt.title("Histogram of Weights for layer{} before Pruning".format(idx+1))
            plt.savefig('layer{} before pruning.png'.format(idx+1))
        else:
            plt.title("Histogram of Weights for layer{} after Pruning".format(idx+1))
            plt.savefig('layer{} after pruning.png'.format(idx+1))
        plt.close()


def main():
    args = parse_arguments()

    # Dataset
    dataset = Dataset(**vars(args))

    # Tensorflow 2.0 by default uses Eager-Execution, hence Placeholders are not getting executed
    tf.compat.v1.disable_eager_execution()

    # Reset the default graph and set a graph-level seed
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(9)

    # Model
    model = Model(num_classes=dataset.num_classes, **vars(args))
    model.construct_model()

    # Session
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    tf.compat.v1.local_variables_initializer().run()

    # Prune
    sparsity_fraction = prune.prune(args, model, sess, dataset)

    # Train and test
    train.train(args, model, sess, dataset)
    test.test(args, model, sess, dataset)

    sess.close()
    sys.exit()


if __name__ == "__main__":
    main()
