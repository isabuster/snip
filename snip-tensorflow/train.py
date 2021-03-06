import os
import tensorflow as tf
import time
import numpy as np

from augment import augment


def train(args, model, sess, dataset, lr, rewinding_itr1=10000, rewinding_itr2=10000):
    print('|========= START TRAINING =========|')
    if not os.path.isdir(args.path_summary): os.makedirs(args.path_summary)
    if not os.path.isdir(args.path_model): os.makedirs(args.path_model)
    saver = tf.compat.v1.train.Saver(max_to_keep=10)
    random_state = np.random.RandomState(9)
    writer = {}
    writer['train'] = tf.compat.v1.summary.FileWriter(args.path_summary + '/train', sess.graph)
    writer['val'] = tf.compat.v1.summary.FileWriter(args.path_summary + '/val')
    t_start = time.time()

    for itr in range(args.train_iterations):
        batch = dataset.get_next_batch('train', args.batch_size)
        batch = augment(batch, args.aug_kinds, random_state)
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
        feed_dict.update({model.compress: False, model.new_compress: False, model.is_train: True, model.pruned: True})
        feed_dict.update({model.lr: lr})
        input_tensors = [model.outputs] # always execute the graph outputs
        if (itr+1) % args.check_interval == 0:
            input_tensors.extend([model.summ_op, model.sparsity])
        input_tensors.extend([model.train_op, model.w_final])
        result = sess.run(input_tensors, feed_dict)

        # Check on validation set.
        if (itr+1) % args.check_interval == 0:
            batch = dataset.get_next_batch('val', args.batch_size)
            batch = augment(batch, args.aug_kinds, random_state)
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
            feed_dict.update({model.compress: False, model.new_compress: False, model.is_train: False, model.pruned: True})
            input_tensors = [model.outputs, model.summ_op, model.sparsity]
            result_val = sess.run(input_tensors, feed_dict)

        # Check summary and print results
        if (itr+1) % args.check_interval == 0:
            writer['train'].add_summary(result[1], itr)
            writer['val'].add_summary(result_val[1], itr)
            pstr = '(train/val) los:{:.3f}/{:.3f} acc:{:.3f}/{:.3f} spa:{:.3f}'.format(
                result[0]['los'], result_val[0]['los'],
                result[0]['acc'], result_val[0]['acc'],
                result[2],
            )
            print('itr{}: {} (t:{:.1f})'.format(itr+1, pstr, time.time() - t_start))
            t_start = time.time()

        # Save model
        if (itr+1) % args.save_interval == 0:
            saver.save(sess, args.path_model + '/itr-' + str(itr))

        # Save weights for rewinding
        if (itr+1) == rewinding_itr1:
            rewinding_weights1 = result[-1]
        if (itr+1) == rewinding_itr2:
            rewinding_weights2 = result[-1]

    return rewinding_weights1, rewinding_weights2
