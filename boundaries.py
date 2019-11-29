import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags
import torch
from torch.autograd import Variable

import calculate_kernels as ck
import dataset
from cnn import MNIST_arch_0
from inference import predict_probs
from do_attack import EmpiricalPrior, verify_params, initialize_kernel, initialize_Kxx_inverse

f = flags.FLAGS

def main(_):
    FLAGS = flags.FLAGS
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
        print("Directory " , FLAGS.output_dir ,  " Created ")

    num_features = 28*28*1
    num_dirs = FLAGS.num_directions
    
    direction_file = os.path.join(FLAGS.output_dir, FLAGS.direction_file.format(FLAGS.num_directions))
    if os.path.exists(direction_file):
        #Test direction file's directions are the right amount:
        directions = np.load(direction_file)
        assert(directions.shape == (num_dirs, num_features))
    else:
        #Generate directions:
        randmat = np.random.normal(size=(num_features,  num_dirs))
        q, r = np.linalg.qr(randmat)
        assert(q.shape == (num_features, num_dirs))
        directions = q.T * np.sqrt(float(num_features)) # gxr3 operates in RMS error
        np.save(direction_file, directions)
    
    directions = np.concatenate([directions, -directions])

    X, Y, _, _, Xt, Yt = dataset.mnist_sevens_vs_twos(FLAGS.data_path, noisy=True)
    

    if FLAGS.model == 'gp':

        params = EmpiricalPrior(FLAGS.seed)
        params = verify_params(params)

        #Create the GP
        with tf.device("GPU:0"):
            model = ck.create_kern(params)

        #Kxx = initialize_kernel("Kxx", X, None, False, model, FLAGS.gp_dir)
        K_inv = initialize_Kxx_inverse(FLAGS.gp_dir)
        K_inv_Y = K_inv @ Y

        def classify(X_pert, pert, index):
            #Don't use initialize_kernel function, because we don't really want to cache these
            Kxpx = ck.calculate_K("Kxpx_{}_{}".format(pert, index), X_pert, X, False, model).squeeze()
            class_probs = predict_probs(Kxpx, K_inv_Y)
            Y_pred_n = np.argmax(class_probs, 1)
            return Y_pred_n

    elif FLAGS.model == 'cnn':
        model = torch.load(FLAGS.cnn_dir, map_location=torch.device('cpu'))
        def classify(X_pert, pert, index):
            X_prep = torch.from_numpy(X_pert.astype(np.float32))
            X_prep = X_prep.reshape(-1, 1, 28, 28)
            xs = Variable(X_prep)
            #if torch.cuda.is_available():
            #    xs = xs.cuda()
            preds = model(xs)
            preds_np = preds.data.cpu().numpy()
            return np.argmax(preds_np, axis=1)
    else:
        raise Exception('Model not supported')

    scaling = [0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 50.0]
    flip_counts = np.zeros(len(scaling)+1)

    #indices = np.random.randint(0, Xt.shape[0], FLAGS.num_patterns) #[0,1,2,3,4]; #
    if FLAGS.indices_file is not None:
        indices = np.loadtxt(FLAGS.indices_file).astype(int)[0:FLAGS.num_patterns]
    else:
        indices = np.arange(0,Xt.shape[0])
        np.random.shuffle(indices)
        np.savetxt('{}/indices.txt'.format(FLAGS.output_dir,FLAGS.num_patterns), indices, fmt='%i')
        indices = indices[0:FLAGS.num_patterns]

    #Also visualize a direction for each pattern
    sess = tf.Session()
    single_adv_x_op = tf.placeholder(tf.float64, Xt[0].shape)
    encode_op = tf.image.encode_png(tf.reshape(tf.cast(single_adv_x_op*255, tf.uint8), (28, 28, 1)))

    for index in indices:
        img = Xt[index]
        y = Yt[index]
        #Expand correct class into big matrix for comparison with all perturbed images at once
        flip_cmp = np.full(num_dirs * 2, np.argmax(y), dtype=np.float32)

        active_dirs = np.arange(0, num_dirs*2)
        #flip_dists = np.full(num_dirs * 2, np.inf, dtype=np.float32) #Distance at which classification flips

        previously_flipped = 0
        for i, s in enumerate(scaling):
            #Create perturbed images
            imgs = img[None] + s * directions[active_dirs]
            imgs = np.clip(imgs, 0.0, 1.0)

            enc_img = sess.run(encode_op, feed_dict={single_adv_x_op: imgs[0]}) #Only one image in a batch
            f = open('{}/pat_{}_scale_{}_dir_{}.png'.format(FLAGS.output_dir, index, s, 0), "wb+")
            f.write(enc_img)
            f.close()

            #Classify perturbed images
            preds = classify(imgs, s, index)

            #If perturbed image switched classes, record that and remove from active_dirs
            still_correct = preds == flip_cmp
            now_flipped = preds != flip_cmp
            #stopped_dirs = active_dirs[now_flipped]
            #flip_dists[stopped_dirs] = s

            still_correct_count = np.count_nonzero(still_correct)
            incorrect_count = now_flipped.sum()
            flip_counts[i] += (incorrect_count - previously_flipped)
            previously_flipped = incorrect_count
            print(still_correct_count)
            if still_correct_count == 0:
                break

            #Then repeat for bigger scaling

        flip_counts[len(scaling)] += still_correct_count

    import pdb; pdb.set_trace()
    flip_counts = flip_counts / float(len(indices) * num_dirs*2)
    #Do something with this information.

    plt.bar(np.arange(0, len(scaling)+1), flip_counts)
    labels = [None]*(len(scaling) + 1)
    for i, s in enumerate(scaling):
        labels[i] = "{}".format(s)
    labels[len(scaling)] = ' > {}'.format(scaling[-1])
    plt.xticks(np.arange(0, len(scaling)+1), labels=labels)
    plt.savefig("{}/{}_flip_dists_norm_({}).png".format(FLAGS.output_dir, FLAGS.model, FLAGS.num_patterns), format='png')
    plt.close()




if __name__ == '__main__':
    flags.DEFINE_integer('seed', 20, 'The seed to use')

    flags.DEFINE_string('data_path', '/scratch/etv21/conv_gp_data/MNIST_data/expA2', "Path to the compressed dataset")
    flags.DEFINE_string('output_dir', '/scratch/etv21/boundary_dists', "Location where generated files will be placed (graphs, kernels, etc)")

    flags.DEFINE_string('cnn_dir', '/scratch/etv21/conv_gp_data/MNIST_data/expA2/cnn_7_vs_2.pt', "Directory where the cnn's .pt file is")
    flags.DEFINE_string('gp_dir', '/scratch/etv21/conv_gp_data/expA1/kernels', "Directory of the GP's kernels")
    flags.DEFINE_enum('model', 'cnn', ['gp', 'cnn'], 'The attack strategy to use. Only specify if generating attack.')


    flags.DEFINE_string('indices_file', '/scratch/etv21/conv_gp_data/expA1/shuffled_indices.txt', "Directory of shuffled test set indices. The first num_patterns many of these will be used.")
    flags.DEFINE_integer('num_patterns', 200, 'How many patterns to look at')

    flags.DEFINE_integer('num_directions', 28*28, 'Number of directions to explore')
    flags.DEFINE_string('direction_file', 'directions_{}.npy', 'File where exploration directions are saved for re-use')
    tf.app.run()