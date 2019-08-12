
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
def summarize_error(K_zz, K_z_x, Y_z, K_inv, K_inv_Y, key, path, make_plots=True):
    print("Calculating", key)

    descrip = key
    Y_pred = K_z_x @ K_inv_Y

    #
    if len(Y_pred.shape) == 1 or Y_pred.shape[1] == 1:
        print("**I'm not sure why we're in this case**")
        Y_pred_n = np.ravel(Y_pred >= 0)
    else:
        Y_pred = Y_pred.squeeze()
        class_probs = softmax(Y_pred)
        #import pdb; pdb.set_trace()
        Y_pred_n = np.argmax(class_probs, 1)
    
    t = sklearn.metrics.accuracy_score(np.argmax(Y_z, 1), Y_pred_n)
    print(key, 'error: ',(1-t)*100,'%')
    f = open(os.path.join(path, "error.txt"),"w+")
    f.write("{} error: {} %".format(key,(1-t)*100,'%'))
    f.close()
    

    if make_plots:
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


'''
def conjugate_inference():
    print("Centering labels")
    Y[Y == 0.] = -1
    K_inv_y = np.array(Y, dtype=FLAGS.float_type, copy=True, order='F')

    print("Size of Kxx, Y, Kxvx, Yv, Kxtx, Yt, Kxax, Ya:", Kxx.shape, Y.shape, Kxvx.shape, Yv.shape, Kxtx.shape, Yt.shape, Kxax.shape, Ya.shape)
    make_symm(Kxx)

    K_inv_path = os.path.join(kernels_path, "K_inv{:02d}_.npy".format(seed))
    if os.path.isfile(K_inv_path):
        print("Loading inverse from file")
        K_inv = np.load(K_inv_path)
    else:
        print("Calculating inverse")
        K_inv = scipy.linalg.inv(Kxx.astype(np.float64), check_finite=True)
    
        identity = np.identity(K_inv.shape[0], dtype=np.float64)
        err = np.abs((K_inv @ Kxx) - identity).sum()
        if err > 0.0005:
            print('WARNING: Large inversion error {}'.format(err))
        np.save(K_inv_path, K_inv)

    K_inv_y = K_inv @ Y

    #Solve A x = b for x, 
    # In this case, we're solving Kxx * (something) = Y where (something) = (Kxx^-1)* Y
    #The input below is a little confusing because (for optimization reasons) we're passing in K_inv_y, which at this point is actually equal to Y 
    #K_inv_y_control = scipy.linalg.solve(Kxx, K_inv_y, overwrite_a=True, overwrite_b=False, check_finite=True, assume_a='pos')
    # magma.posv(Kxx.T, K_inv_y, lower=True)
    #But now, we also want the variance. So we just calculate Kxx inverse directly so we can use it in the variance computations.
    params = dict(depth=32)
    params['id'] = 'seed_{}_arch_{}'.format(seed, a)
    # print_error(Kxx, K_inv_y, Y,  params, "training_error") 
    #print_error(Kxvx,  K_inv_y, Yv, params, "validation_error")
    #print_error(Kxtx,  K_inv_y, Yt, params, "test_error")
    #print_error(Kxax,  K_inv_y, Ya, params, "adversarial_error")
    #results(Kxx, K_inv_y, Y,  params, "training_error") 
    results(Kaa, Kxax, Ya, K_inv, K_inv_y, params, 'adversarial', output_dir, make_plots=True)
    
    results(Kvv, Kxvx, Yv, K_inv, K_inv_y, params, 'validation', output_dir, make_plots=(a == 0))
    results(Ktt, Kxtx, Yt, K_inv, K_inv_y, params, 'test', output_dir, make_plots=(a == 0))
    params['time'] = time.time() - start_time
    df = pd.DataFrame(data=params, index=pd.Index([0]))
    df.to_csv(csv_file)
    print(df)



def main(_):
    start_time = time.time()
    FLAGS = flags.FLAGS

    print("Loading kernel matrices")
    data_path = FLAGS.data_dir
    adv_data_file = FLAGS.adv_data_file

    print("Looking for non-adversarial data in "+data_path)
    #For architectures 0 - 8
    for a in range(0,1):
        seed = FLAGS.seed
        np.random.seed(seed)
        
        #Different matrices come from different experiments
        kernels_path = os.path.join(FLAGS.kernel_dir, "kernels")
        adv_kernels_path = os.path.join(FLAGS.adv_kernel_dir, "kernels")
        print("Loading kernels from {} \n\t and adversarial kernels from {}".format(kernels_path, adv_kernels_path))

        #exp0 matrices
        Kxx = load_array(os.path.join(kernels_path, "Kxx_{:02d}.npy".format(seed)))
        Kxtx = load_array(os.path.join(kernels_path, "Kxtx_{:02d}.npy".format(seed)))
        Kxvx = load_array(os.path.join(kernels_path, "Kxvx_{:02d}.npy".format(seed)))
        #exp1 matrices
        Ktt = load_array(os.path.join(kernels_path, "Kt_diag{:02d}.npy".format(seed)))
        Kvv = load_array(os.path.join(kernels_path, "Kv_diag{:02d}.npy".format(seed)))
        #Dependent on the adv attacks:
        Kxax = load_array(os.path.join(adv_kernels_path, "Kxax_{:02d}_arch_{}.npy".format(seed, a)))
        Kaa = load_array(os.path.join(adv_kernels_path, "Ka_diag{:02d}_arch_{}.npy".format(seed, a)))

        output_dir = os.path.join(FLAGS.adv_kernel_dir, FLAGS.output_dir)

        csv_file = os.path.join(output_dir, "results_{:02d}.csv".format(FLAGS.layer_i))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("CSV file is:", csv_file)

        print("Loading data")

        #If using the original code, then here we should be using save_kernels instead.
        #But we want to use save_kernel_simple
        _, Y, _, Yv, _, Yt =  save_kernel_simple.mnist_sevens_vs_twos(data_path)
        #cut_training(FLAGS.N_train, FLAGS.N_vali,*getattr(save_kernels, FLAGS.dataset)())
        #adv_data_path = os.path.join(data_path, 'eps=0.3_arch_{}'.format(a))
        _, Ya = save_kernel_simple.mnist_sevens_vs_twos_adv(data_path,  data_file=adv_data_file)

        if FLAGS.inference == 'conjugate':
            print("Centering labels")
            Y[Y == 0.] = -1
            K_inv_y = np.array(Y, dtype=FLAGS.float_type, copy=True, order='F')

            print("Size of Kxx, Y, Kxvx, Yv, Kxtx, Yt, Kxax, Ya:", Kxx.shape, Y.shape, Kxvx.shape, Yv.shape, Kxtx.shape, Yt.shape, Kxax.shape, Ya.shape)
            make_symm(Kxx)

            K_inv_path = os.path.join(kernels_path, "K_inv{:02d}_.npy".format(seed))
            if os.path.isfile(K_inv_path):
                print("Loading inverse from file")
                K_inv = np.load(K_inv_path)
            else:
                print("Calculating inverse")
                K_inv = scipy.linalg.inv(Kxx.astype(np.float64), check_finite=True)
            
                identity = np.identity(K_inv.shape[0], dtype=np.float64)
                err = np.abs((K_inv @ Kxx) - identity).sum()
                if err > 0.0005:
                    print('WARNING: Large inversion error {}'.format(err))
                np.save(K_inv_path, K_inv)

            K_inv_y = K_inv @ Y

            #Solve A x = b for x, 
            # In this case, we're solving Kxx * (something) = Y where (something) = (Kxx^-1)* Y
            #The input below is a little confusing because (for optimization reasons) we're passing in K_inv_y, which at this point is actually equal to Y 
            #K_inv_y_control = scipy.linalg.solve(Kxx, K_inv_y, overwrite_a=True, overwrite_b=False, check_finite=True, assume_a='pos')
            # magma.posv(Kxx.T, K_inv_y, lower=True)
            #But now, we also want the variance. So we just calculate Kxx inverse directly so we can use it in the variance computations.
            #import pdb; pdb.set_trace();
        else:
            assert FLAGS.float_type == 'float32'
            #magma.make_symm(Kxx)
            Kxx = make_symm(Kxx)
            # Divide kernel by something sensible because its eigenvalues are huge
            Kxx /= np.mean(Kxx)
            #Approximate posterior
            ap = ncgp.AP(Kxx)
            # Classifier will output >0 if label is 1 and <0 if label is 0
            lik = ncgp.SigmoidClassifierLikelihood(Y[:, 1])

            fun = {'nc_laplace': lik.laplace,
                'nc_vi': lik.vi,
                'nc_ep': lik.ep}[FLAGS.inference]
            for i in tqdm.trange(FLAGS.n_iter):
                # Relevant things we could implement: (NOTE(adria): outdated but I'll leave it here)
                # Opper and Winther: "we therefore choose to make a greedy update of lambda_i only every 20th iteration step"
                # loo estimate of generalization error (by Opper and Winther)
                # We could be using line search but in the experiments of Salimbeni, Eleftheriadis and Hensman, the optimal step size is close to 1 anyways
                ap = fun(ap, alpha=1)
                if i % 10 == 0:
                    K_inv_y = np.ravel(ap.invK_mu)
                    print_error(Kxvx, K_inv_y, Yv, {}, 'val')
            K_inv_y = np.ravel(ap.invK_mu)

        params = dict(depth=32)
        params['id'] = 'seed_{}_arch_{}'.format(seed, a)
        # print_error(Kxx, K_inv_y, Y,  params, "training_error") 
        #print_error(Kxvx,  K_inv_y, Yv, params, "validation_error")
        #print_error(Kxtx,  K_inv_y, Yt, params, "test_error")
        #print_error(Kxax,  K_inv_y, Ya, params, "adversarial_error")
        #results(Kxx, K_inv_y, Y,  params, "training_error") 
        results(Kaa, Kxax, Ya, K_inv, K_inv_y, params, 'adversarial', output_dir, make_plots=True)
        
        results(Kvv, Kxvx, Yv, K_inv, K_inv_y, params, 'validation', output_dir, make_plots=(a == 0))
        results(Ktt, Kxtx, Yt, K_inv, K_inv_y, params, 'test', output_dir, make_plots=(a == 0))
        params['time'] = time.time() - start_time
        df = pd.DataFrame(data=params, index=pd.Index([0]))
        df.to_csv(csv_file)
        print(df)

if __name__ == '__main__':
    f = flags
    #f.DEFINE_enum('dataset', 'mnist_1hot_all', ['mnist_1hot_all', 'cifar_all', 'oz_1hot_all', 'cifar_catdog'], 'the dataset to use')
    f.DEFINE_enum('inference', 'conjugate', ['conjugate', 'nc_laplace', 'nc_vi', 'nc_ep'],
                  'form of likelihood and type of inference')
    f.DEFINE_integer('N_train', 55000, 'number of training data points')
    f.DEFINE_integer('N_vali', 5000, 'number of validation data points')
    f.DEFINE_integer('layer_i', -1, "Layer's kernel to use")
    f.DEFINE_integer('n_gpus', 1, "Number of GPUs to use")

    f.DEFINE_string('kernel_dir', '/scratch/etv21/conv_gp_data/exp4/exp4_E2',
                    "the path to the precomputed kernel matrices, with subdirectory `kernels` containing the kernels")
    f.DEFINE_string('adv_kernel_dir', '/scratch/etv21/conv_gp_data/exp4/exp4_E2/inf',
                    "the path to the precomputed adversarial kernel matrices, with subdirectory `kernels` containing the kernels")
    f.DEFINE_string('data_dir', '/scratch/etv21/conv_gp_data/MNIST_data/eps=0.3_arch_0/',
                    "the path to the raw data")
    f.DEFINE_string('adv_data_file', '../reverse/gp_adversarial_examples_noisy_eps=0.3_norm_inf.npy',
                    "the path to the adversarial data, relative to normal data path") #'two_vs_seven_adversarial_noisy_eps=0.3.npy'
    f.DEFINE_integer('seed', 20, "Random seed")
    f.DEFINE_string('output_dir', '',
                    "directory to save plots and CSVs with results, under `FLAGS.adv_kernel_dir`")
    f.DEFINE_enum('float_type', 'float32', ['float32', 'float64'], 'Float type to operate with')
    f.DEFINE_integer('n_iter', 1000, 'number of iterations for natural gradient descent')
    absl_app.run(main)

    # export LD_PRELOAD="/homes/ag919/intel/mkl/lib/intel64_lin/libmkl_intel_thread.so:/homes/ag919/intel/mkl/lib/intel64_lin/libmkl_core.so:/usr/lib/gcc/x86_64-linux-gnu/5/libgomp.so:${LD_PRELOAD}"
    # export LD_LIBRARY_PATH="/homes/ag919/intel/mkl/lib/intel64_lin:${LD_LIBRARY_PATH}"

    # python3 classify_gp.py --dataset=oz_1hot_all --kernel_file=/scratch/ag919/grams_resnet/oz_01.h5 --n_gpus=1 --N_train=10604 --N_vali=2061
'''