import numpy as np
import tensorflow as tf
import os
import os.path as path
import sys
import scipy.linalg as slin
import pickle_utils as pu

import dataset
import attacks
import calculate_kernels as ck
import inference as results


def SimpleMNISTParams(seed):
    return dict(
            seed=seed,
            var_weight=[0.642928 for k in range (0,3)], #np.random.rand() * 8 + 0.5, #Don't really know what to choose for weights and biases. Same as init for the conv? 0.0004682
            var_bias=[3.761496 for k in range(0,3)], #np.random.rand() * 8 + 0.2, 0.002308
            n_layers=2,
            filter_sizes=5,
            strides=1,
            padding="SAME", 
            nlin="ExReLU",  
            skip_freq=-1,
    )

def verify_params(gp_params):
    if gp_params['skip_freq'] > 0:
        gp_params['padding'] = 'SAME'
        gp_params['strides'] = 1
    
    print("Params:", params)
    return gp_params

def make_symm(A, lower=False):
    i_lower = np.tril_indices(A.shape[-1], -1)
    if lower:
        A.T[i_lower] = A[i_lower]
    else:
        A = A.squeeze()
        A[i_lower] = A.T[i_lower]
    return A


def load_array(arr_path):
    if os.path.islink(arr_path):
        arr_path = os.readlink(arr_path)
    return np.load(arr_path).squeeze()

def initialize_kernel(name, X1, X2, diag, kern, kernels_dir):
    kernel_file = path.join(kernels_dir, name + '.npy')
    #Try to load the kernel.
    print('Trying to load kernel {} from {}'.format(name, kernel_file))
    if os.path.isfile(kernel_file):
        print("Loading {} from file".format(name))
        return np.load(kernel_file)
    else:
        #If it doesn't exist, calculate and save it
        Kx1x2 = ck.calculate_K(name, X1, X2, diag, kern).squeeze()
        np.save(kernel_file, Kx1x2, allow_pickle=False)
        return Kx1x2

def initialize_Kxx_inverse(kernels_dir, safety_check=True):
    #Get K_inv_Y
    K_inv_path = os.path.join(kernels_dir, "K_inv.npy")
    #Load it if we've already calculated it, otherwise actually calculate it
    if os.path.isfile(K_inv_path):
        print("Loading inverse from file")
        K_inv = np.load(K_inv_path)
    else:
        print("Calculating inverse")
        Kxx = np.load(os.path.join(kernels_dir, "Kxx.npy")).squeeze()
        make_symm(Kxx)
        K_inv = slin.inv(Kxx.astype(np.float64), check_finite=True)
        if safety_check:
            identity = np.identity(K_inv.shape[0], dtype=np.float64)
            err = np.abs((K_inv @ Kxx) - identity).sum()
            if err > 0.0005:
                print('WARNING: Large inversion error {}'.format(err))
        np.save(K_inv_path, K_inv)
    return K_inv

def classify(key, Xz, Yz, z, X, K_inv, K_inv_Y, kern, kernels_dir, output_dir):

    Kxzx = initialize_kernel("Kx{}x".format(z), Xz, X, False, kern, kernels_dir)
    Kz_diag = initialize_kernel("K{}_diag".format(z), Xz, None, True, kern, kernels_dir)

    print('Shape of ',"Kx{}x".format(z),' is {}'.format(Kxzx.shape))
    print('Shape of ',"K{}_diag".format(z),' is {}'.format(Kz_diag.shape))
    results.summarize_error(Kz_diag, Kxzx, Yz, K_inv, K_inv_Y, key, output_dir)


if __name__ == '__main__':
    seed = 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 20
    n_gpus = 1

    param_constructor = SimpleMNISTParams
    epsilon=0.3
    norm_type=np.Inf

    data_path = '/scratch/etv21/conv_gp_data/MNIST_data/eps=0.3_arch_0'
    output_dir = '/scratch/etv21/conv_gp_data/exp4/exp4_E2'
    
    attack_dir = path.join('/scratch/etv21/conv_gp_data/exp4/exp4_F3','eps={}_norm_{}'.format(epsilon, norm_type))
    adv_dir = '/scratch/etv21/conv_gp_data/MNIST_data/reverse/'
    #Filename if attack is being generated and will be saved (as this filename)
    adv_file_output ='gp_adversarial_examples_eps={}_norm_{}'.format(epsilon, norm_type)
    generate_attack = True
    #Filename if using attack that has already been generated.
    adv_data_file = 'gp_adversarial_examples_eps=0.5_norm_inf.npy' #'two_vs_seven_adversarial.npy' # 
    
    np.random.seed(seed)
    tf.set_random_seed(seed)

    kernels_dir = os.path.join(output_dir, "kernels")
    if not os.path.exists(kernels_dir):
        os.makedirs(kernels_dir)
        print("Directory " , kernels_dir ,  " Created ")

    #Load all the data (train, test, val)
    X, Y, Xv, Yv, Xt, Yt = dataset.mnist_sevens_vs_twos(data_path, noisy=True)

    #Parameters for the GP
    params = param_constructor(seed)
    params = verify_params(params)

    pu.dump(params, path.join(output_dir, 'params.pkl.gz'))
    #Create the GP
    with tf.device("GPU:0"):
        kern = ck.create_kern(params)

    #Calculate the training kernel and its inverse, if it doesn't exist.
    #If it already exists, just load it.
    #We do classification by treating it as a regression problem i.e. the conjugate method
    #So all we need is the inverse of the training kernel
    Kxx = initialize_kernel("Kxx", X, None, False, kern, kernels_dir)
    K_inv = initialize_Kxx_inverse(kernels_dir)
    #Center labels and make symmetric:
    Y[Y == 0.] = -1
    K_inv_Y = K_inv @ Y

    classify('test', Xt, Yt, 't', X, K_inv, K_inv_Y, kern, kernels_dir, output_dir)
    classify('validation', Xv, Yv, 'v', X, K_inv, K_inv_Y, kern, kernels_dir, output_dir)

    adv_kernels_dir = os.path.join(attack_dir, "kernels")
    if not os.path.exists(adv_kernels_dir):
        os.makedirs(adv_kernels_dir)
        print("Directory " , adv_kernels_dir ,  " Created ")

    #So at this point, we no longer need any of the kernels, just the inverse of the training. 
    #Generate attack and save adversarial examples
    if generate_attack:
        print('Generating attack')
        #Technically, we should be using Y_t_pred to avoid leaking
        Yt_adv = np.copy(Yt)
        Yt_adv[Yt_adv == 0.] = -1
        Xa = attacks.fgsm(K_inv_Y, kern, X, Xt, Yt_adv, seed=seed, epsilon=epsilon, output_images=True, max_output=50, norm_type=norm_type, output_path=adv_dir, adv_file_output=adv_file_output)
    else:
        print('Loading attack')
        Xa = np.load(path.join(adv_dir, adv_data_file))
        #Xa = Xa.reshape(-1, 28*28)

    #Calculate adversarial kernels and error
    classify('adv', Xa, Yt, 'a', X, K_inv, K_inv_Y, kern, adv_kernels_dir, attack_dir)




def PaperParams():
    return dict(
            seed=53,
            var_weight=7.273299, 
            var_bias=4.689324, 
            n_layers=9,
            filter_sizes=4,
            strides=1,
            padding="SAME", 
            nlin="ExReLU", 
            skip_freq=1,
    )


#Exp 5 C
def LayerMatchedMNISTParams1(seed):
    return dict(
            seed=seed,
            var_weight=[0.01598815806210041*32.0, 0.0008328557014465332*64.0, 0.00015443310257978737* 64 * 28*28], #Matches CNN layers' empirical variance * number of channels in conv layer (or inputs in FC layer)
            var_bias=[0.006102397572249174*32.0, 0.0003918512666132301*64.0,  0.00000491933406010503* 64 * 28*28],
            n_layers=2,
            filter_sizes=5,
            strides=1,
            padding="SAME", 
            nlin="ExReLU",  
            skip_freq=-1,
    )

#Exp 5 D
def LayerMatchedMNISTParams2(seed):
    return dict(
            seed=seed,
            var_weight=[0.01598815806210041*32.0, 0.0008328557014465332*64.0, 0.00015443310257978737* 64], #Matches CNN layers' empirical variance * number of channels in conv layer (64 in FC layer)
            var_bias=[0.006102397572249174*32.0, 0.0003918512666132301*64.0,  0.00000491933406010503* 64],
            n_layers=2,
            filter_sizes=5,
            strides=1,
            padding="SAME", 
            nlin="ExReLU",  
            skip_freq=-1,
    )

#Exp 5 E
def LayerMatchedMNISTParams3(seed):
    return dict(
        seed=seed,
        var_weight=[0.01598815806210041*32.0, 0.0008328557014465332*64.0, 0.00015443310257978737], #Matches CNN layers' empirical variance * number of channels in conv layer (1 in FC layer)
        var_bias=[0.006102397572249174*32.0, 0.0003918512666132301*64.0,  0.00000491933406010503],
        n_layers=2,
        filter_sizes=5,
        strides=1,
        padding="SAME", 
        nlin="ExReLU",  
        skip_freq=-1,
    )

def RandomSearchParams(seed):
    return dict(
            seed=seed,
            var_weight=np.random.rand() * 8 + 0.5,
            var_bias=np.random.rand() * 8 + 0.2,
            n_layers=4 + int(np.random.rand()*12),
            filter_sizes=3 + int(np.random.rand()*4),
            strides= 1+ int(np.random.rand()*3),
            padding=("VALID" if np.random.rand() > 0.5 else "SAME"),
            nlin=("ExReLU" if np.random.rand() > 0.5 else "ExErf"),
            skip_freq=(int(np.random.rand()*2) + 1 if np.random.rand() > 0.5 else -1), 
    )
