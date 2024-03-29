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


from absl import app as absl_app
from absl import flags

#From Adria's paper
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

def EmpiricalPrior(seed):
    return dict(
            seed=seed,
            var_weight=[0.0428030751645565*25, 0.0015892550582066178*800, 0.0001530303416075185*50176], 
            var_bias=[0.032435204833745956*25, 0.0012280623195692897*800, (2.100331403198652e-05)*50176],
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
    
    print("Params:", gp_params)
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

def remove_kernels(z, kernels_dir):
    diag_kernel_file = path.join(kernels_dir, "K{}_diag.npy".format(z))
    if os.path.isfile(diag_kernel_file):
        os.remove(diag_kernel_file) 
    cov_kernel_file = path.join(kernels_dir, "Kx{}x.npy".format(z))
    if os.path.isfile(cov_kernel_file):
        os.remove(cov_kernel_file) 

    
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

def main(_):
    FLAGS = flags.FLAGS

    if FLAGS.param_constructor == 'EmpiricalPrior':
        param_constructor = EmpiricalPrior
    elif FLAGS.param_constructor == 'SimpleMNISTParams':
        param_constructor = SimpleMNISTParams
    elif FLAGS.param_constructor == 'PaperParams':
        param_constructor = PaperParams
    else:
        print('Unsupported parameter struct specified')
        return
    attack_name = FLAGS.adv_attack_name.format(FLAGS.epsilon, FLAGS.norm_type)
    #Directory where the adv kernels and adv-specific graphs will be go
    adv_output_dir = os.path.join(FLAGS.adv_output, attack_name)
    #Directory where the adv dataset is/will be
    adv_data_dir = os.path.join(FLAGS.adv_data, attack_name)
    #Filename if attack is being generated and will be saved (as this filename), or of the file to be loaded if the attack already exists
    adv_data_file = FLAGS.adv_data_file.format(attack_name)
    
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    kernels_dir = os.path.join(FLAGS.output_dir, "kernels")
    if not os.path.exists(kernels_dir):
        os.makedirs(kernels_dir)
        print("Directory " , kernels_dir ,  " Created ")

    #Load all the data (train, test, val)
    X, Y, Xv, Yv, Xt, Yt = dataset.mnist_sevens_vs_twos(FLAGS.data_path, noisy=True)

    #Parameters for the GP
    params = param_constructor(FLAGS.seed)
    params = verify_params(params)

    pu.dump(params, path.join(FLAGS.output_dir, 'params.pkl.gz'))
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
    #Don't center labels. Use one-hot vectors as probabilities
    #Y[Y == 0.] = -1
    K_inv_Y = K_inv @ Y

    if not FLAGS.adv_only:
        classify('test', Xt, Yt, 't', X, K_inv, K_inv_Y, kern, kernels_dir, FLAGS.output_dir)
        classify('validation', Xv, Yv, 'v', X, K_inv, K_inv_Y, kern, kernels_dir, FLAGS.output_dir)
    
    adv_kernels_dir = os.path.join(adv_output_dir, "kernels")
    if not os.path.exists(adv_kernels_dir):
        os.makedirs(adv_kernels_dir)
        print("Directory " , adv_kernels_dir ,  " Created ")
    
    #So at this point, we no longer need any of the kernels, just the inverse of the training. 
    #Generate attack and save adversarial examples
    if FLAGS.generate_attack:
        print('Generating attack')
        #Yt_adv = np.copy(Yt)
        #Yt_adv[Yt_adv == 0.] = -1
        remove_kernels('a', adv_kernels_dir)

        if FLAGS.attack == 'fgsm':
            Xa = attacks.fgsm(K_inv_Y, kern, X, Xt, Yt, seed=FLAGS.seed, epsilon=FLAGS.epsilon, norm_type=FLAGS.norm_type, output_images=True, max_output=128, output_path=adv_data_dir, adv_file_output=adv_data_file)
        else:
            Xa = attacks.fgsm_cleverhans(K_inv_Y, kern, X, Xt, Yt, epsilon=FLAGS.epsilon, norm_type=FLAGS.norm_type, output_images=True, max_output=128,  output_path=adv_data_dir, adv_file_output=adv_data_file)
    else:
        print('Loading attack')
        Xa = np.load(path.join(adv_data_dir, adv_data_file))
        #Xa = Xa.reshape(-1, 28*28)

    #Calculate adversarial kernels and error
    classify('adv', Xa, Yt, 'a', X, K_inv, K_inv_Y, kern, adv_kernels_dir, adv_output_dir)

if __name__ == '__main__':
    f = flags
    f.DEFINE_integer('seed', 20, 'The seed to use')
    f.DEFINE_enum('param_constructor', 'EmpiricalPrior', ['EmpiricalPrior', 'SimpleMNISTParams', 'PaperParams'], 'The GP parameter struct to use')
    f.DEFINE_float('epsilon', 0.3, 'The FGSM perturbation size')
    f.DEFINE_float('norm_type', np.Inf, 'The norm to be used by FGSM')

    f.DEFINE_string('data_path', '/scratch/etv21/conv_gp_data/MNIST_data/expA',
                    "Path to the compressed dataset")
    f.DEFINE_string('output_dir', '/scratch/etv21/conv_gp_data/expA1',
                    "Location where all generated files will be placed (graphs, kernels, etc)")

    f.DEFINE_enum('attack', 'fgsm', ['fgsm', 'cleverhans_fgsm'], 'The attack strategy to use')

    f.DEFINE_string('adv_attack_name', 'GP_FGSM_eps={}_norm_{}_nll_targeted',
                    "Name of attack. All outputs related to this attack will be put in the adv_output/adv_data directories, within a new subdirectory with this (adv_attack_name) name")

    f.DEFINE_string('adv_output', '/scratch/etv21/conv_gp_data/expA1/',
                    "Directory where the adv kernels will be put. Usually a subdirectory of the output_dir")

    f.DEFINE_string('adv_data', '/scratch/etv21/conv_gp_data/MNIST_data/expA/',
                    "Directory where the adv dataset is/will be")

    f.DEFINE_string('adv_data_file', 'two_vs_seven_{}.npy',
                    "Filename of attack data. If the is being generated, the new data will be saved as this filename. Otherwise, the name of file to be loaded if the attack already exists. Final name will be <value>.format(attack_name)")

    f.DEFINE_bool('generate_attack', True,
                    "Whether the attack is generated, or whether an existing attack should be loaded")
    f.DEFINE_bool('adv_only', True, 
                    "When this flag is true, the test and validation error won't be evaluated. Useful when generating a bunch of different attacks for the same GP")
    absl_app.run(main)


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
