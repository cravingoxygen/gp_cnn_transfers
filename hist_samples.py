

from absl import app as absl_app
from absl import flags
import os
import numpy as np
from cnn import extract_images_and_labels
import numpy.linalg as lin
import dataset
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def main(_=None):
    X, Y, Xv, Yv, Xt, Yt = dataset.mnist_sevens_vs_twos(FLAGS.data_path, noisy=True)
    Xa_cnn, Xa_gp = np.load(FLAGS.adv_path_cnn), np.load(FLAGS.adv_path_gp)
    ignore_list_2s = [35, 998, 1015]
    ignore_list_7s = [102, 126, 153, 914, 915, 916, 969, 1032]
    start_2s = 0
    start_7s = 1032
    twos_clean = np.delete(Xt[:start_7s], ignore_list_2s, axis=0)
    twos_adv_cnn = np.delete(Xa_cnn[:start_7s], ignore_list_2s, axis=0).reshape(-1, 28*28)
    twos_adv_gp = np.delete(Xa_gp[:start_7s], ignore_list_2s, axis=0)
    sevens_clean = np.delete(Xt[start_7s:], ignore_list_7s, axis=0)
    sevens_adv_cnn = np.delete(Xa_cnn[start_7s:], ignore_list_7s, axis=0).reshape(-1, 28*28)
    sevens_adv_gp = np.delete(Xa_gp[start_7s:], ignore_list_7s, axis=0)

    #import pdb; pdb.set_trace()

    pixel_xs = np.arange(0,28*28)

    plt.subplot(2, 3, 1)
    plt.bar(pixel_xs, twos_clean.sum(axis=0)) #/twos_clean.shape[0]
    plt.title('Clean 2s')
    plt.ylabel('Intensity')
    plt.xlabel('Pixel')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])

    plt.subplot(2, 3, 2)
    plt.bar(pixel_xs, twos_adv_cnn.sum(axis=0))
    plt.title('CNN Adversarial 2s')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])


    plt.subplot(2, 3, 3)
    plt.bar(pixel_xs, twos_adv_gp.sum(axis=0))
    plt.title('GP Adversarial 2s')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])

    plt.subplot(2, 3, 4)
    plt.bar(pixel_xs, sevens_clean.sum(axis=0)) #/sevens_clean.shape[0]
    plt.title('Clean 7s')
    plt.ylabel('Intensity')
    plt.xlabel('Pixel')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])

    plt.subplot(2, 3, 5)
    plt.bar(pixel_xs, sevens_adv_cnn.sum(axis=0))
    plt.title('CNN Adversarial 7s')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])


    plt.subplot(2, 3, 6)
    plt.bar(pixel_xs, sevens_adv_gp.sum(axis=0))
    plt.title('GP Adversarial 7s')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    axes = plt.gca()
    axes.set_xlim([0,28*28])
    axes.set_ylim([0,800])

    plt.show()


    


if __name__ == '__main__':
    flags.DEFINE_string('adv_path_gp', '/scratch/etv21/conv_gp_data/MNIST_data/expA/cleverhans_FGSM_eps=0.08_norm_inf/two_vs_seven_cleverhans_FGSM_eps=0.08_norm_inf.npy', 'Path to adversarial samples')
    flags.DEFINE_string('adv_path_cnn', '/media/etv21/Elements/CNN_Attacks/attacks/two_vs_seven_adv_CNN_FGSM_eps=0.08_norm=inf.npy', 'Path to adversarial samples')
    flags.DEFINE_string('data_path', '/scratch/etv21/conv_gp_data/MNIST_data/expA2', 'Dir of clean MNIST datafiles')
    flags.DEFINE_string('output', '/scratch/etv21/histograms', 'Dir for placing generated files')

    absl_app.run(main)

