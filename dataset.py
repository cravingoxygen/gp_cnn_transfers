import numpy as np
import os
import tensorflow as tf

def extract_images_and_labels(class_one, class_two, images, labels):
    classes = np.argmax(labels, 1)
    class_one_mask = classes == class_one
    class_two_mask = classes == class_two
    extracted_images = np.concatenate((images[class_one_mask], images[class_two_mask]))
    extracted_labels = np.concatenate( (np.array([[1,0]]*np.sum(class_one_mask)), np.array([[0,1]]*np.sum(class_two_mask))) )
    return (extracted_images, extracted_labels)


def mnist_sevens_vs_twos_adv(data_path, data_file='two_vs_seven_adversarial.npy', labels_file='two_vs_seven_labels.npy'):
    
    adv_images = np.load(os.path.join(data_path, data_file)).squeeze()
    adv_images = adv_images.reshape(-1, 28*28)
    #These labels aren't one hot
    adv_labels = np.load(os.path.join(data_path, labels_file)).squeeze()
    #Make them one hot:
    num_sevens = np.sum(adv_labels)
    num_twos = adv_labels.shape[0] - num_sevens
    adv_one_hot_labels = np.concatenate( (np.array([[1,0]]*num_twos), np.array([[0,1]]*num_sevens)) )
    
    '''
    #Shuffle within each class for sanity check
    indices_twos = np.load('perm_twos_test.npy')
    indices_sevens = np.load('perm_sevens_test.npy') + num_twos
    indices = np.concatenate((indices_twos, indices_sevens))
    return adv_images[indices], adv_one_hot_labels[indices]
    '''
    return adv_images, adv_one_hot_labels

def mnist_sevens_vs_twos(data_path, noisy=True, float_type=np.float64):
    from tensorflow.examples.tutorials.mnist import input_data
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    d = input_data.read_data_sets(data_path, one_hot=True)

    train_images, train_labels = extract_images_and_labels(2, 7, d.train.images, d.train.labels)
    test_images, test_labels = extract_images_and_labels(2, 7, d.test.images, d.test.labels) #mnist_sevens_vs_twos_adv(data_path, data_file='two_vs_seven_test.npy')
    validation_images, validation_labels = extract_images_and_labels(2, 7, d.validation.images, d.validation.labels)

    if noisy:
        train_noise = np.load(data_path + '/X_noise_20.npy')
        test_noise = np.load(data_path + '/Xt_noise_20.npy')
        validation_noise = np.load(data_path + '/Xv_noise_20.npy')

        train_images = train_images + train_noise
        test_images = test_images + test_noise
        validation_images = validation_images + validation_noise

    r = tuple(a.astype(float_type) for a in [
        train_images, train_labels,
        validation_images, validation_labels,
        test_images, test_labels])
    tf.logging.set_verbosity(old_v)
    return r