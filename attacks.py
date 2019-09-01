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

import cleverhans.model
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, ProjectedGradientDescent, ElasticNetMethod
#from tensorflow.nn import softmax_cross_entropy_with_logits_v2
#from tensorflow.losses import softmax_cross_entropy

def PGD_Params(eps=0.3, eps_iter=0.05, ord=np.Inf, nb_iter=10, rand_init=True, clip_grad=True):
	return dict(
		eps=eps,
		eps_iter=eps_iter,
		ord=ord,
		nb_iter=nb_iter,
		rand_init=rand_init,
		clip_grad=clip_grad,
		clip_min=np.float64(0.0),
		clip_max=np.float64(1.0),
	)

def EAD_Params(beta=1e-2, confidence=0, decision_rule='EN', learning_rate=5e-3, binary_search_steps=6, max_iterations=500, initial_const=1e-2,abort_early=True,batch_size=1):
	return dict(
		beta=beta,
		decision_rule=decision_rule,
		batch_size=batch_size,
		confidence=confidence,
		learning_rate=learning_rate,
		binary_search_steps=binary_search_steps,
		max_iterations=max_iterations,
		abort_early=abort_early,
		initial_const=initial_const,
		clip_min=np.float64(0.0),
		clip_max=np.float64(1.0),
	)

def FGSM_Params(eps, ord):
    return dict(
            eps=eps,
            ord=ord,
            clip_min=np.float64(0.0),
            clip_max=np.float64(1.0),
    )

def CW_L2_Params(confidence=0, learning_rate=5e-3, binary_search_steps=6, max_iterations=500, initial_const=1e-2,abort_early=True, batch_size=1):
    return dict(
            batch_size=batch_size,
            confidence=confidence,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iterations=max_iterations,
            abort_early=abort_early,
            initial_const=initial_const,
            clip_min=np.float64(0.0),
            clip_max=np.float64(1.0),
    )

def attack(attack_method, attack_params, K_inv_Y, kernel, X, Xt, Yt, output_path, adv_file_output, output_images=True, max_output=128):
    print(attack_params)
    if 'batch_size' in attack_params:
        batch_size = attack_params['batch_size']
    else:
        batch_size = 1

    #Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_images_dir = os.path.join(output_path, '{}_images'.format(adv_file_output))
    if output_images and not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)
    sess = gpflow.get_default_session()

    K_inv_Y_ph = tf.constant(K_inv_Y)
    X2_ph = tf.constant(X)
    #K_inv_Y_ph = tf.placeholder(settings.float_type, K_inv_Y.shape, 'K_inv_Y')
    #X2_ph = tf.placeholder(settings.float_type, X.shape, 'X_train')
    #Callable that returns logits
    def predict_callable(xt):
        Kxtx_op = kernel.K(xt, X2_ph)
        predict_op = tf.matmul(Kxtx_op, K_inv_Y_ph)
        return predict_op
    #Convert callable to model
    model = cleverhans.model.CallableModelWrapper(predict_callable, 'logits')
    #Define attack part of graph
    if (attack_method == 'fgsm'):
        attack_obj = FastGradientMethod(model, sess=sess, dtypestr='float64')
    elif (attack_method == 'cw_l2'):
        attack_obj = CarliniWagnerL2(model, sess=sess, dtypestr='float64')
    elif (attack_method == 'ead'):
        attack_obj = ElasticNetMethod(model, sess=sess, dtypestr='float64')
    elif (attack_method == 'pgd'):
        attack_obj = ProjectedGradientDescent(model, sess=sess, dtypestr='float64')
        
    x = tf.placeholder(settings.float_type, shape=(None, Xt.shape[1]))
    adv_x_op = attack_obj.generate(x, **attack_params)
    preds_adv_op = model.get_logits(adv_x_op)
    
    adv_examples = None
    batch_num = 0
    for k in tqdm.trange(0, Yt.shape[0], batch_size):
        end = min(k + batch_size , Yt.shape[0])
        #feed_dict = {K_inv_Y_ph: K_inv_Y, X_ph: Xt[k:end, :], X2_ph: X}
        #import pdb; pdb.set_trace()
        feed_dict = { x: Xt[k:end, :]}
        yt = Yt[k:end, :]
        adv_x, preds_adv = sess.run((adv_x_op, preds_adv_op), feed_dict=feed_dict)
        
        if adv_examples is None:
            adv_examples = np.array(adv_x.reshape(batch_size, 28*28))
        else:
            adv_examples = np.append(adv_examples, adv_x.reshape(batch_size, 28*28), 0)

        if output_images and (max_output == None or max_output > batch_num * batch_size or (batch_num*batch_size >= 1280 and batch_num*batch_size < 1280 + max_output)):
            for c in range(0, batch_size):
                adv_img = adv_x[c]*255
                adv_img = (adv_img.astype(int)).reshape(28,28)
                scipy.misc.toimage(adv_img, cmin=0, cmax=255).save(path.join(output_images_dir, 'gp_attack_{}_noisy.png'.format(batch_num*batch_size + c)))
        batch_num += 1

    np.save(os.path.join(output_path, adv_file_output), adv_examples, allow_pickle=False)
    
    return adv_examples


def fgsm_cleverhans(K_inv_Y, kernel, X, Xt, Yt, epsilon=0.3, norm_type=np.Inf, output_images=True, max_output=128, output_path='/scratch/etv21/conv_gp_data/MNIST_data/cleverhans_fgsm/',  adv_file_output='cleverhans_fgsm_eps={}_norm_{}'):

    batch_size = 1
    #Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_images_dir = os.path.join(output_path, '{}_images'.format(adv_file_output))
    if output_images and not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)
    sess = gpflow.get_default_session()

    fgsm_params = {'eps': epsilon,'ord': norm_type, 'clip_min': np.float64(0.0), 'clip_max': np.float64(1.0)}

    #Placeholders
    K_inv_Y_ph = tf.placeholder(settings.float_type, K_inv_Y.shape, 'K_inv_Y')
    X2_ph = tf.placeholder(settings.float_type, X.shape, 'X_train')
    #Callable that returns logits
    def predict_callable(xt):
        Kxtx_op = kernel.K(xt, X2_ph)
        predict_op = tf.matmul(Kxtx_op, K_inv_Y_ph)
        return predict_op
    #Convert callable to model
    model = cleverhans.model.CallableModelWrapper(predict_callable, 'logits')
    #Define attack part of graph
    fgsm = FastGradientMethod(model, sess=sess, dtypestr='float64') 
    x = tf.placeholder(settings.float_type, shape=(None, Xt.shape[1]))
    adv_x_op = fgsm.generate(x, **fgsm_params)
    preds_adv_op = model.get_logits(adv_x_op)

    adv_examples = None
    batch_num = 0
    for k in tqdm.trange(0, Yt.shape[0], batch_size):
        end = min(k + batch_size , Yt.shape[0])
        #feed_dict = {K_inv_Y_ph: K_inv_Y, X_ph: Xt[k:end, :], X2_ph: X}
        feed_dict = {K_inv_Y_ph: K_inv_Y, x: Xt[k:end, :], X2_ph: X}
        yt = Yt[k:end, :]
        adv_x, preds_adv = sess.run((adv_x_op, preds_adv_op), feed_dict=feed_dict)
        
        if adv_examples is None:
            adv_examples = np.array(adv_x.reshape(batch_size, 28*28))
        else:
            adv_examples = np.append(adv_examples, adv_x.reshape(batch_size, 28*28), 0)

        if output_images and (max_output == None or max_output > batch_num * batch_size or (batch_num*batch_size >= 1280 and batch_num*batch_size < 1280 + max_output)):
            for c in range(0, batch_size):
                adv_img = adv_x[c]*255
                adv_img = (adv_img.astype(int)).reshape(28,28)
                scipy.misc.toimage(adv_img, cmin=0, cmax=255).save(path.join(output_images_dir, 'gp_attack_{}_noisy.png'.format(batch_num*batch_size + c)))
        batch_num += 1

    np.save(os.path.join(output_path,(adv_file_output + '.npy').format(epsilon, norm_type)), adv_examples, allow_pickle=False)
    return adv_examples

def fgsm(K_inv_Y, kernel, X, Xt, Yt, seed=20,
    epsilon=0.3, clip_min=0.0, clip_max=1.0, norm_type='inf', batch_size=1, 
    output_images=False, max_output=None, output_path='/scratch/etv21/conv_gp_data/MNIST_data/reverse/',  adv_file_output='gp_adversarial_examples_eps={}_norm_{}'):
    
    #Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_images_dir = os.path.join(output_path, '{}_images'.format(adv_file_output))
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

        #For plain MSE loss
        #-tf.losses.mean_squared_error(predict_op, -Yt_ph)
        
        #For negative log likelihood:
        #We can't use the tf negative log likelihood function because it requires doubles, which our graphics card doesn't support.
        #So the function we're trying to implement is:
        #   - (1/n) sum(  log(y_i)  ) 
        #Where y_i is the is the probability of pattern i being assigned its true class. By maximing this, we are by the virtues of probability,
        # also minimizing the probability of y_i being assigned to the wrong class.

        #Multiply predictions by one-hot labels to only get predictions of the true class. Then take the log softmax and sum over the batch.
        #Do we need to divide by the batch_size? It's a constant, so it shouldn't matter.
        Ytarget = 1 - Yt_ph
        tmp_softmax = tf.nn.softmax(predict_op)
        loss = tf.reduce_sum(tf.math.multiply(tf.math.log(tmp_softmax), Ytarget))

        #For MSE of softmax loss
        #loss = tf.losses.mean_squared_error(tf.nn.softmax(predict_op),Yt_ph)
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
        elif norm_type == 1:
            abs_grad = tf.stop_gradient(tf.math.abs(grad))
            max_abs_grad = tf.reduce_max(abs_grad, axis=1, keepdims=True)
            tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
            num_ties = tf.reduce_sum(tied_for_max, axis=1, keepdims=True)
            
            sign_grad = tf.math.sign(grad)
            optimal_perturbation = sign_grad * tied_for_max / num_ties

        elif norm_type == 2:
            const_not_zero = tf.constant(1e-12, dtype=tf.float64)
            square = tf.maximum(const_not_zero, tf.reduce_sum(tf.square(grad),axis=1,keepdims=True))
            optimal_perturbation = grad / tf.sqrt(square)


        scaled_perturbation = tf.math.scalar_mul(eps, optimal_perturbation)

        X_adv_op = X_ph + scaled_perturbation
        X_adv_op = tf.clip_by_value(X_adv_op, clip_min, clip_max)
    
    #sess.graph.finalize()
    #writer = tf.summary.FileWriter(kernel_path + '/tboard', sess.graph)

    batch_num = 0
    for k in tqdm.trange(0, Yt.shape[0], batch_size):
        end = min(k + batch_size , Yt.shape[0])
        feed_dict = {K_inv_Y_ph: K_inv_Y, Yt_ph: Yt[k:end, :], X_ph: Xt[k:end, :], X2_ph: X}
        #adv_example_batch, grad_res = sess.run((Kxtx_op,grad), feed_dict=feed_dict)
        #adv_example_batch, batch_predictions, batch_loss, batch_grad, batch_grad_sign, batch_opt_pert = sess.run((X_adv_op, predict_op, loss, grad, grad_sign, optimal_perturbation), feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
        adv_example_batch = sess.run((X_adv_op), feed_dict=feed_dict)
        if adv_examples is None:
            adv_examples = np.array(adv_example_batch.reshape(batch_size, 28*28))
        else:
            adv_examples = np.append(adv_examples, adv_example_batch.reshape(batch_size, 28*28), 0)

        if output_images and (max_output == None or max_output > batch_num * batch_size or (batch_num*batch_size >= 1280 and batch_num*batch_size < 1280 + max_output)):
            for c in range(0, batch_size):
                adv_img = adv_example_batch[c]*255
                adv_img = (adv_img.astype(int)).reshape(28,28)
                scipy.misc.toimage(adv_img, cmin=0, cmax=255).save(path.join(output_images_dir, 'gp_attack_{}_noisy.png'.format(batch_num*batch_size + c)))
        batch_num += 1

    np.save(os.path.join(output_path,(adv_file_output + '.npy').format(epsilon, norm_type)), adv_examples, allow_pickle=False)
    return adv_examples
