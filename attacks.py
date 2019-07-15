"""
Compute deep kernels with a bunch of different hyperparameters and save them to
disk
"""
import numpy as np
import tensorflow as tf
import sys
from gpflow import settings

from os import path
#sys.path.append(path.abspath('~/Python/github.com/cambridge-mlg'))

#from Project1.file1 import something
import ncgp

import tqdm
import pickle_utils as pu
import os
import gpflow
import h5py
import scipy


def fgsm(K_inv_Y, kernel, X, Xt, Yt, seed=20,
    epsilon=0.3, clip_min=0.0, clip_max=1.0, norm_type='inf', batch_size=5, 
    output_images=False, max_output=None, output_path='/scratch/etv21/conv_gp_data/MNIST_data/reverse/',  adv_file_output='gp_adversarial_examples_eps={}_norm_{}'):
    
    #Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_images_dir = os.path.join(output_path, 'images_eps_{}_norm_{}'.format(epsilon, norm_type))
    if output_images and not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)


    adv_examples = None
    sess = gpflow.get_default_session()
    with tf.device("gpu:0"):
        print("Xt's shape: {}   X's shape: {}".format(Xt.shape, X.shape))
        print("K_inv_Y's shape: {}".format(K_inv_Y.shape))
        print("Yt's shape: {}".format(Yt.shape))
        
        K_inv_Y_ph = tf.placeholder(settings.float_type, K_inv_Y.shape, 'K_inv_Y')
        Yt_ph = tf.placeholder(settings.float_type, [None, Yt.shape[1]], 'Y_test')

        X_ph = tf.placeholder(settings.float_type, [batch_size, Xt.shape[1]], 'X_test')
        X2_ph = tf.placeholder(settings.float_type, X.shape, 'X_train')
        Kxtx_op = kernel.K(X_ph, X2_ph)
        
        predict_op = tf.matmul(Kxtx_op, K_inv_Y_ph)
        loss = tf.losses.mean_squared_error(predict_op, Yt_ph)
        grad = tf.gradients(loss, X_ph, stop_gradients=[K_inv_Y_ph, Yt_ph, X2_ph])[0]
        #Check for Nan/Infinite gradients
        with tf.control_dependencies([tf.debugging.assert_all_finite(grad,'grad is not well-defined')]):
            grad = tf.identity(grad)

        eps = tf.constant(epsilon, dtype=settings.float_type)
        if norm_type == np.inf:
            #L-inf norm for epsilon:
            abs_grad = tf.stop_gradient(tf.math.abs(grad))
            #Find elements of the gradient with abs value larger than the scale of the noise we add
            const_bound = tf.constant(0.0001, dtype=tf.float64)
            mask = tf.math.greater(abs_grad,const_bound)
            #Only perturb gradient elements that are large enough
            large_enough_grad = tf.multiply(grad, tf.cast(mask, tf.float64))
            optimal_perturbation = tf.sign(large_enough_grad)
        elif norm_type == 2:
            const_not_zero = tf.constant(1e-12, dtype=tf.float64)
            square = tf.maximum(const_not_zero, tf.reduce_sum(tf.square(grad),axis=1,keepdims=True))
            optimal_perturbation = grad / tf.sqrt(square)


        scaled_perturbation = tf.math.scalar_mul(eps, optimal_perturbation)

        X_adv_op = X_ph + scaled_perturbation
        X_adv_op = tf.clip_by_value(X_adv_op, clip_min, clip_max)
    
    sess.graph.finalize()
    #writer = tf.summary.FileWriter(kernel_path + '/tboard', sess.graph)

    batch_num = 0
    for k in tqdm.trange(0, Yt.shape[0], batch_size):
        end = min(k + batch_size , Yt.shape[0])
        feed_dict = {K_inv_Y_ph: K_inv_Y, Yt_ph: Yt[k:end, :], X_ph: Xt[k:end, :], X2_ph: X}
        #adv_example_batch, grad_res = sess.run((Kxtx_op,grad), feed_dict=feed_dict)
        #adv_example_batch, batch_predictions, batch_loss, batch_grad, batch_grad_sign, batch_opt_pert = sess.run((X_adv_op, predict_op, loss, grad, grad_sign, optimal_perturbation), feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
        adv_example_batch = sess.run((X_adv_op), feed_dict=feed_dict)
        
        if adv_examples is None:
            adv_examples = np.array(adv_example_batch.reshape(batch_size, 28, 28))
        else:
            adv_examples = np.append(adv_examples, adv_example_batch.reshape(batch_size, 28, 28), 0)

        if output_images and (max_output == None or max_output > batch_num * batch_size):
            for c in range(0, batch_size):
                adv_img = adv_example_batch[c]*255
                adv_img = (adv_img.astype(int)).reshape(28,28)
                scipy.misc.toimage(adv_img, cmin=0, cmax=255).save(path.join(output_images_dir, 'gp_attack_{}_noisy.png'.format(batch_num*batch_size + c)))
        batch_num += 1

    np.save(os.path.join(output_path,adv_file_output.format(epsilon, norm_type)), adv_examples, allow_pickle=False)
    return adv_examples

'''
if __name__ == '__main__':
    seed = 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 20
    
    n_gpus = 1
    root_path = "/scratch/etv21/conv_gp_data/"
    #Path to where the existing kernels are
    exp_path = os.path.join(root_path, "exp4/exp4_E2")
    #Path to the MNIST dataset
    data_path = os.path.join(root_path,'MNIST_data/eps=0.3_arch_0')
    print('Data path: ' + data_path)

    kernel_path = os.path.join(exp_path, "kernels")
    if not os.path.exists(path):
        print("'kernels' directory not found at " , exp_path )

    np.random.seed(seed)
    tf.set_random_seed(seed)

    X, Y, _, _, Xt, Yt = save_kernel_simple.mnist_sevens_vs_twos(data_path)
    K_inv = training_kernel_inverse(kernel_path, seed)
    K_inv_Y = K_inv @ Y
    kern = save_kernel_simple.create_kern(save_kernel_simple.SimpleMNISTParams(seed))
    print("Generating attack:")
    Xa = fgsm(K_inv_Y, kern, X, Xt, Yt, seed=seed, epsilon=0.3, output_images=True, max_output=50, norm_type=np.Inf)

    
'''