
"""
Using a precomputed kernel matrix, compute test scores of GP regression
"""
import numpy as np
import scipy
import sklearn.metrics
import time
import tqdm
import os

import sys
import deep_ckern

from absl import flags
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax

def entropy(probs):
    return  -(probs[:, 0] * np.log(probs[:,0])) -(probs[:, 1] * np.log(probs[:,1]))

def make_symm(A, lower=False):
    i_lower = np.tril_indices(A.shape[-1], -1)
    if lower:
        A.T[i_lower] = A[i_lower]
    else:
        A = A.squeeze()
        A[i_lower] = A.T[i_lower]
    return A

def save_pattern(Xt, index, filename):
    pattern = Xt[index]*255
    adv_img = (pattern.astype(int)).reshape(28,28)
    scipy.misc.toimage(adv_img, cmin=0, cmax=255).save(filename)

def print_error(K_xt_x, K_inv_y, Ytv, dit, key):
    print("Calculating", key)
    Y_pred = K_xt_x @ K_inv_y
    if len(Y_pred.shape) == 1 or Y_pred.shape[1] == 1:
        Y_pred_n = np.ravel(Y_pred >= 0)
    else:
        Y_pred_n = np.argmax(Y_pred, 1)
        t = sklearn.metrics.accuracy_score(
        np.argmax(Ytv, 1), Y_pred_n)
    dit[key] = (1-t)*100
    print(dit)

#K_zz = The diagonal of the test inputs
#K_z_x = Symmetric covariance between test inputs and training inputs
#Y_z = test outputs
#K_inv = Inverted covariance matrix from training
#K_inv_Y = K_inv * Y_train, sent in to avoid repeated calculation
def summarize_error(K_zz, K_z_x, Y_z, K_inv, K_inv_Y, key, path, make_plots=True, set_name=None):
    print("Calculating", key)
    if set_name == None:
        set_name = key
    descrip = set_name
    Y_pred = K_z_x @ K_inv_Y

    #
    if len(Y_pred.shape) == 1 or Y_pred.shape[1] == 1:
        print("**I'm not sure why we're in this case**")
        Y_pred_n = np.ravel(Y_pred >= 0)
    else:
        Y_pred = Y_pred.squeeze()
        class_probs = softmax(Y_pred)
        entropies = entropy(class_probs)
        #import pdb; pdb.set_trace()
        Y_pred_n = np.argmax(class_probs, 1)
    
    t = sklearn.metrics.accuracy_score(np.argmax(Y_z, 1), Y_pred_n)
    print(set_name, 'error: ',(1-t)*100,'%')
    f = open(os.path.join(path, "error_report.txt"),"a+")
    f.write("{} error: {} %".format(set_name,(1-t)*100))
    f.close()
    

    if make_plots:
        plt.figure(figsize=(12, 9), dpi=100)
        plt.hist(entropies, bins=100, density=True)
        plt.savefig("{}/entropy_{}.png".format(path, descrip), format='png')
        axes = plt.gca()
        plt.close()

        variance = K_zz - np.diagonal((K_z_x @ K_inv @ K_z_x.T))
        variance_accuracy_plots(variance, np.argmax(Y_z, 1), Y_pred_n, path, descrip)

        plt.figure(figsize=(12, 9), dpi=100)
        plt.hist(variance, bins=100, density=True)
        plt.savefig("{}/variance_{}.png".format(path, descrip), format='png')
        axes = plt.gca()
        axes.set_xlim([0,1500])
        plt.close()

        
        #Y_pred[0] = 1 - Y_pred[1] because we're applying softmax, so we only bother plotting one
        plt.figure(figsize=(12, 9), dpi=100)
        plt.scatter(range(0,class_probs.shape[0]), class_probs[:,0], alpha=0.3, c=variance, cmap='YlOrRd') 
        plt.colorbar()
        axes = plt.gca()
        #axes.set_ylim([-1.5,1.5])
        plt.savefig("{}/prediction_{}.png".format(path, descrip), format='png')
        plt.close()
        
#THe predictions received as input here should be the actual class predictions, already processed to be one-hot instead of real numbers
def variance_accuracy_plots(variance, Y_pred, Y, path, descrip):
    correct_mask = (Y_pred - Y) == 0
    incorrect_mask = np.logical_not(correct_mask)
    plt.hist([variance[correct_mask], variance[incorrect_mask]], bins=100, stacked=True)
    plt.legend( ['Correct', "Incorrect"])
    axes = plt.gca()
    #axes.set_xlim([0,1.0e-7])
    axes.set_ylim([0,90])
    
    plt.savefig("{}/variance_acc_{}.png".format(path, descrip), format='png')
    plt.close()


def cut_training(N_train, N_vali, X, Y, Xv, Yv, Xt, Yt):
    """
    If you computed a kernel matrix larger than the size of the training set
    you want to use, this is useful!
    For example: the default TensorFlow MNIST training set is 55k examples, but
    the paper "Deep Neural Networks as Gaussian Processes" by Jaehon Lee et al.
    (https://arxiv.org/abs/1711.00165) used only 50k, and we had to make a fair
    comparison.
    """
    return (X[:N_train], Y[:N_train],
            np.concatenate([X[N_train:], Xv], axis=0)[:N_vali, :],
            np.concatenate([Y[N_train:], Yv], axis=0)[:N_vali, :],
            Xt, Yt)

def load_array(path):
    if os.path.islink(path):
        path = os.readlink(path)
    return np.load(path).squeeze()