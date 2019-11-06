import pandas as pd
from tensorflow.python.platform import flags
import os
import numpy as np
from cnn import extract_images_and_labels
import tensorflow as tf
import numpy.linalg as lin
import dataset as ds

FLAGS = flags.FLAGS

DTYPE_STR = 'float32'

if DTYPE_STR == 'float64':
	FLOAT_TYPE = np.float64
	TF_FLOAT_TYPE=tf.float64
else:
	FLOAT_TYPE = np.float32
	TF_FLOAT_TYPE=tf.float32

def draw_patterns(xs, output_dir, start_2s=0, start_7s=1280, num_patterns=128, image_dir_name='images'):
	output_image_path = os.path.join(output_dir, image_dir_name)
	print('Images will be saved to: {}'.format(output_image_path))
	if not os.path.exists(output_image_path):
		os.makedirs(output_image_path)
	
	sess = tf.Session()
	single_adv_x_op = tf.placeholder(TF_FLOAT_TYPE, shape=(28,28))
	encode_op = tf.image.encode_png(tf.reshape(tf.cast(single_adv_x_op*255, tf.uint8), (28, 28, 1)))
	
	start_2s = min(xs.shape[0], max(start_2s, 0))
	end_2s = min(xs.shape[0],  start_2s+num_patterns)
	for c in range(start_2s,end_2s):
		enc_img = sess.run(encode_op, feed_dict={single_adv_x_op: xs[c]}) #Only one image in a batch
		f = open(output_image_path + '/{}.png'.format(c), "wb+")
		f.write(enc_img)
		f.close()
		
	if start_7s is not None:
		start_7s = min(xs.shape[0], max(start_7s, 0))
		end_7s = min(xs.shape[0],  start_7s+num_patterns)
		for c in range(start_7s,end_7s):
			enc_img = sess.run(encode_op, feed_dict={single_adv_x_op: xs[c]}) #Only one image in a batch
			f = open(output_image_path + '/{}.png'.format(c), "wb+")
			f.write(enc_img)
			f.close()
	
#two_vs_seven_adv_CNN_cw_l2_conf=0.0_max_iter=50_init_c=1.0_lr=0.05
def infer_attack_name(file_name):
	attack_name = file_name[len('two_vs_seven_adv_'):]
	attack_name = attack_name[0:len(attack_name) - len('.npy')]
	return attack_name
	
	
def calc_dists(test_images, xs, attack_name, report_file):
	#import pdb; pdb.set_trace()
	distance = (test_images.reshape(-1, 28*28) - xs.reshape(-1, 28*28))
			
	inf_dist = lin.norm(distance, ord=np.Inf, axis=1)
	l1_dist = lin.norm(distance, ord=1, axis=1)
	l2_dist = lin.norm(distance, ord=2, axis=1)
			
	if report_file is not None:
		f = open(report_file, "a+")
		f.write('{}\tave \t std \t max \t min\n'.format(attack_name))
		f.write('L2    \t {}\t {}\t {} {}\n'.format(l2_dist.mean(), l2_dist.std(), l2_dist.max(), l2_dist.min()))
		f.write('L1    \t {}\t {}\t {} {}\n'.format(l1_dist.mean(), l1_dist.std(), l1_dist.max(), l1_dist.min()))
		f.write('LInf  \t {}\t {}\t {} {}\n'.format(inf_dist.mean(), inf_dist.std(), inf_dist.max(), inf_dist.min()))
		f.close()

def main(_=None):
	#Draw sample patterns of all the datasets in a directory
	from tensorflow.examples.tutorials.mnist import input_data
	ds = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	test_images, test_labels = extract_images_and_labels(2, 7, ds.test.images, ds.test.labels)
	test_noise = np.random.normal(0, 0.0001, test_images.shape)
	
	test_images = test_images + test_noise
	np.clip(test_images, 0.0, 1.0, out=test_images)
	
	for file in os.listdir(FLAGS.parent_dir):
		if file.endswith(".npy") and file.startswith('two_vs_seven_adv_'):
			patterns = np.load(os.path.join(FLAGS.parent_dir, file))
			attack_name = infer_attack_name(file)
			print(attack_name)
			calc_dists(test_images, patterns, attack_name, FLAGS.report_file)
			draw_patterns(patterns, os.path.join(FLAGS.parent_dir, attack_name))


if __name__ == '__main__':
	'''
	
	'/home/squishymage/cnn_gp/training_data'
	'/home/squishymage/cnn_gp/training_data'
	'/home/squishymage/cnn_gp/training_data/report.txt'

	'''
	flags.DEFINE_string('parent_dir', '/home/squishymage/cnn_gp/training_data', 'The directory containing the .npy files')
	flags.DEFINE_string('data_dir', '/home/squishymage/cnn_gp/training_data', 'Dir of clean MNIST datafiles') 
	flags.DEFINE_string('report_file', '/home/squishymage/cnn_gp/training_data/report.txt', 'File for writing dists')
	#flags.DEFINE_boolean'check_subdirs', False, 'File with labels for adversarial data. Should be inside the data_dir')

	tf.app.run()
