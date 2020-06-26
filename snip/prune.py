import time
import numpy as np

def prune_snip(args, model, sess, dataset):
    print('|========= START SNIP =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.compress: True, model.new_compress: False, model.is_train: False, model.pruned: False})
    result = sess.run([model.outputs, model.sparsity, model.num_weights, model.sparsity_fraction], feed_dict)
    # print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[1], time.time() - t_start))
    print("Sparsity per layer:")
    print(result[-1])
    return result[-2], result[-1]


def prune_magnitude(args, model, sess, dataset, kappa):
    print('|========= START MAGNITUDE PRUNING =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.kappa[k]: kappa[k] for k in kappa})
    feed_dict.update({model.compress: False, model.new_compress: True, model.is_train: True, model.pruned: True})
    result = sess.run([model.outputs, model.sparsity, model.w_final], feed_dict)
    print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[1], time.time() - t_start))

def rewind(args, model, sess, dataset, rewinding_weights, rewinding_itr=4000):
    print('Rewinding weights to itr-{}'.format(rewinding_itr))
    for k in rewinding_weights:
        assign_op = model.net.weights_ap[k].assign(rewinding_weights[k])
        sess.run(assign_op)
