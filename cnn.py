"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.linalg as lin
import tensorflow as tf
from tensorflow.python.platform import flags
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, ProjectedGradientDescent, ElasticNetMethod, SPSA
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from attacks import CW_L2_Params, EAD_Params, PGD_Params, FGSM_Params, SPSA_Params

import matplotlib.pyplot as plt
from PIL import Image 
import pickle
import tqdm
import math
import time
import alternative_archs as archs
#import draw_samples as ds

FLAGS = flags.FLAGS

DTYPE_STR = 'float32'

if DTYPE_STR == 'float64':
	FLOAT_TYPE = np.float64
	TF_FLOAT_TYPE=tf.float64
else:
	FLOAT_TYPE = np.float32
	TF_FLOAT_TYPE=tf.float32
NB_EPOCHS = 6
BATCH_SIZE = 103
LEARNING_RATE = .001

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED) 

tf.set_random_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_images_and_labels(class_one, class_two, images, labels):
	classes = np.argmax(labels, 1)
	class_one_mask = classes == class_one
	class_two_mask = classes == class_two
	extracted_images = np.concatenate((images[class_one_mask], images[class_two_mask]))
	#extracted_labels = np.concatenate((labels[class_one_mask], labels[class_two_mask]))
	extracted_labels = np.concatenate( (np.array([0]*np.sum(class_one_mask)), np.array([1]*np.sum(class_two_mask))) )
	return (extracted_images, extracted_labels)

def mnist_sevens_vs_twos(data_path, noisy=True, float_type=FLOAT_TYPE): 
	from tensorflow.examples.tutorials.mnist import input_data
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)
	ds = input_data.read_data_sets(data_path, one_hot=True)
	
	train_images, train_labels = extract_images_and_labels(2, 7, ds.train.images, ds.train.labels)
	test_images, test_labels = extract_images_and_labels(2, 7, ds.test.images, ds.test.labels)
	#validation_images, validation_labels = extract_images_and_labels(2, 7, ds.validation.images, ds.validation.labels)
	
	if noisy:
		train_noise = np.load(data_path + '/X_noise_20.npy') #np.random.normal(0, 0.0001, train_images.shape)
		test_noise = np.load(data_path + '/Xt_noise_20.npy') #np.random.normal(0, 0.0001, test_images.shape)
		#validation_noise = np.load(data_path + '/Xv_noise_20.npy')

		train_images = train_images + train_noise
		np.clip(train_images, 0.0, 1.0, out=train_images)
		test_images = test_images + test_noise
		np.clip(test_images, 0.0, 1.0, out=test_images)
		#validation_images = validation_images + validation_noise

	tf.logging.set_verbosity(old_v)
	return MNIST_Dataset(train_images.reshape(-1, 28,28), train_labels), MNIST_Dataset(test_images.reshape(-1, 28,28), test_labels)

class MNIST_Dataset(torch.utils.data.Dataset):
	def __init__(self, images, labels):
		assert images.shape[0] == labels.shape[0]
		self.images = images.reshape(-1, 1, 28, 28).astype(FLOAT_TYPE)
		self.labels = labels.astype(np.long)

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	def __len__(self):
		return self.images.shape[0]

class Adversarial_MNIST_Dataset(torch.utils.data.Dataset):
	def __init__(self, images_path, labels_path):
		images = np.load(images_path).reshape(-1, 28,28)
		labels = np.load(labels_path)
		
		assert images.shape[0] == labels.shape[0]
		self.images = np.expand_dims(images, 1).astype(FLOAT_TYPE)
		self.labels = labels.astype(FLOAT_TYPE)

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	def __len__(self):
		return self.images.shape[0]
	
	
class MNIST_arch_0(nn.Module):
	""" Basic MNIST model from github
	Modified to have one fc layer and no pooling
	https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
	"""

	def __init__(self):
		super(MNIST_arch_0, self).__init__()
		# input is 28x28
		# padding=2 for same padding
		self.conv1 = conv2d_gaussian_init(nn.Conv2d(1, 32, 5, padding=2))
		self.conv2 = conv2d_gaussian_init(nn.Conv2d(32, 64, 5, padding=2))
		# No pooling, so feature map same as input size * num filters
		self.fc1 = fc_gaussian_init(nn.Linear(64 * 28*28, 2))

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 64 * 28*28)  # reshape Variable
		x = self.fc1(x)
		return F.log_softmax(x, dim=-1)


def fc_gaussian_init(fc_layer):
	print('FC layer constant: {}'.format(fc_layer.weight.size(1)))
	stdev = 1./math.sqrt(fc_layer.weight.size(1))
	fc_layer.weight.data.normal_(0, stdev)
	if fc_layer.bias is not None:
		fc_layer.bias.data.normal_(0, stdev)
	return fc_layer

def conv2d_gaussian_init(conv2d_layer):
	n = conv2d_layer.in_channels
	import pdb; pdb.set_trace()
	for k in conv2d_layer.kernel_size:
		n *= k
	print('conv layer constant: {}'.format(n))
	stdev = 1./math.sqrt(n)
	conv2d_layer.weight.data.normal_(0, stdev)
	if conv2d_layer.bias is not None:
		conv2d_layer.bias.data.normal_(0, stdev)
	return conv2d_layer

def transfer_attack(torch_model, attack_name, adv_images_file, labels_file, batch_size=BATCH_SIZE, report_file=None):
	#For transfer from GP examples to CNN:
	adversarial_loader = torch.utils.data.DataLoader(Adversarial_MNIST_Dataset(images_path=adv_images_file, labels_path=labels_file), batch_size=batch_size)

	total = 0
	correct = 0
	
	for xs, ys in adversarial_loader:
		xs, ys = Variable(xs), Variable(ys)
		if torch.cuda.is_available():
			xs, ys = xs.cuda(), ys.cuda()
		
		preds = torch_model(xs)
		preds_np = preds.data.cpu().numpy()

		correct += (np.argmax(preds_np, axis=1) == ys.cpu().numpy()).sum()
		total += len(xs)

	acc = float(correct) / total
	print('Adversarial error: %.2f%%' % ((1 - acc) * 100))
	
	if report_file is not None:
		f = open(report_file, "a+")
		f.write('{}	error: {}%'.format(attack_name, (1 - acc)*100))
		f.close()

	return
	

def train_model(data_dir, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, train_end=-1, test_end=-1, learning_rate=LEARNING_RATE, model_arch=MNIST_arch_0):
	# Train a pytorch MNIST model
	torch_model = model_arch()
	if torch.cuda.is_available():
		torch_model = torch_model.cuda()

	training_dataset, test_dataset = mnist_sevens_vs_twos(data_dir ,noisy=True)
	train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
	
	# Train our model
	optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)

	for _epoch in range(nb_epochs):
		for xs, ys in train_loader:
			xs, ys = Variable(xs), Variable(ys)
			if torch.cuda.is_available():
				xs, ys = xs.cuda(), ys.cuda()
			optimizer.zero_grad()
			preds = torch_model(xs)
			loss = F.nll_loss(preds, ys)
			loss.backward()  # calc gradients
			optimizer.step()  # update gradients
	
	# Evaluate on clean data
	total = 0
	correct = 0
	for xs, ys in test_loader:
		xs, ys = Variable(xs), Variable(ys)
		if torch.cuda.is_available():
			xs, ys = xs.cuda(), ys.cuda()
		preds = torch_model(xs)
		preds_np = preds.data.cpu().numpy()

		correct += (np.argmax(preds_np, axis=1) == ys.cpu().numpy()).sum()
		total += len(xs)

	acc = float(correct) / total
	print('Clean error: %.2f%%' % ((1 - acc) * 100))
	'''
	report_file = os.path.join('/home/squishymage/cnn_gp/archs', FLAGS.report)
	if report_file is not None:
		f = open(report_file, "a+")
		f.write('Clean error: %.2f%%' % ((1 - acc) * 100))
		f.close()
	'''
	return torch_model

def generate_attack(attack, attack_params, torch_model, data_dir, output_dir='', output_samples=True, report_file=None, batch_size=BATCH_SIZE, attack_arch=None):
	time_start = time.time()
	test_dataset = mnist_sevens_vs_twos(data_dir, noisy=True)[1]
	if attack == 'spsa':
		batch_size = 1
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
	
	# Convert pytorch model to a tf_model and wrap it in cleverhans
	tf_model_fn = convert_pytorch_model_to_tf(torch_model)
	cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
	
	# We use tf for evaluation on adversarial data
	sess = tf.Session()
	x_ph = tf.placeholder(TF_FLOAT_TYPE, shape=(None, 1, 28, 28,))
	y_target_ph = tf.placeholder(TF_FLOAT_TYPE, shape=(None))
	if attack == 'fgsm':
		attack_op = FastGradientMethod(cleverhans_model, sess=sess, dtypestr=DTYPE_STR)
		attack_name = 'CNN_FGSM_eps={}_norm={}'.format(attack_params['eps'], attack_params['ord'])
	elif attack == 'pgd':
		attack_op = ProjectedGradientDescent(cleverhans_model, sess=sess, dtypestr=DTYPE_STR)
		attack_name = 'CNN_pgd_eps={}_eps_iter={}_nb_iter={}_ord={}'.format(attack_params['eps'],attack_params['eps_iter'], attack_params['nb_iter'], attack_params['ord'])
	elif attack == 'cw_l2':
		attack_op = CarliniWagnerL2(cleverhans_model, sess=sess, dtypestr=DTYPE_STR)
		attack_name = 'CNN_cw_l2_conf={}_max_iter={}_init_c={}_lr={}'.format(attack_params['confidence'], attack_params['max_iterations'], attack_params['initial_const'], attack_params['learning_rate'])
	elif attack == 'ead':
		attack_op = ElasticNetMethod(cleverhans_model, sess=sess, dtypestr=DTYPE_STR)
		attack_name = 'CNN_ead_beta={}_conf={}_max_iter={}_init_c={}'.format(attack_params['beta'],attack_params['confidence'], attack_params['max_iterations'], attack_params['initial_const'])
	elif attack=='spsa':
		attack_op = SPSA(cleverhans_model, sess=sess, dtypestr=DTYPE_STR)
		attack_name = 'CNN_spsa_eps={}_nb_iter={}_spsa_iters={}_learning_rate={}_delta={}'.format(attack_params['eps'],attack_params['nb_iter'],attack_params['spsa_iters'],attack_params['learning_rate'],attack_params['delta'])
	else:
		print('ERROR - Unsupported attack specified')
		return
		
	if attack_arch is not None:
		attack_name = attack_arch +"_" + attack_name
	
	attack_dir = os.path.join(output_dir, attack_name)
	if not os.path.exists(attack_dir):
		os.makedirs(attack_dir)
	print("Directory " , attack_dir ,  " Created ")
	f = open(os.path.join(attack_dir, "params.txt"),"w+")
	f.write( str(attack_params) )
	f.close()

	adv_x_op = attack_op.generate(x_ph, **attack_params, y_target=y_target_ph)
	adv_preds_op = tf_model_fn(adv_x_op)
	
	single_adv_x_op = tf.placeholder(TF_FLOAT_TYPE, shape=(1,28,28))
	encode_op = tf.image.encode_png(tf.reshape(tf.cast(single_adv_x_op*255, tf.uint8), (28, 28, 1)))

	all_adv_preds = np.array(0)
	adv_images = None
	total = 0
	correct = 0
	c = 0
	first_batch = True
	if output_samples:
		output_image_path = os.path.join(attack_dir, "images")
		print('Images will be saved to: {}'.format(output_image_path))
		if not os.path.exists(output_image_path):
			os.makedirs(output_image_path)
	
	for xs, ys in test_loader:
		#Target is opposite class
		y_targets = 1 + (ys*-1)
		adv_xs, adv_preds = sess.run((adv_x_op,adv_preds_op), feed_dict={x_ph: xs, y_target_ph: y_targets})
		all_adv_preds = np.append(all_adv_preds, adv_preds)
		correct += (np.argmax(adv_preds, axis=1) == ys.cpu().numpy()).sum()
		total += len(xs)
		
		if output_samples and (c < 128 or (c >= 1280 and c < 1280 + 128)):
			if BATCH_SIZE > 1:
				if first_batch:
					print("**For twos***")
				else:
					print("**For sevens**")
					
				distance = (xs.numpy() - adv_xs).reshape(BATCH_SIZE, -1)
				
				inf_dist = lin.norm(distance, ord=np.Inf, axis=1)
				l1_dist = lin.norm(distance, ord=1, axis=1)
				l2_dist = lin.norm(distance, ord=2, axis=1)
				
				if report_file is not None:
					f = open(report_file, "a+")
					if first_batch:
						f.write("\n{} **For twos**\n".format(attack_name))
					else:
						f.write("\n{} **For sevens**\n".format(attack_name))
					f.write('L2 norm - ave: {}\t std: {}\t max: {} min: {}\n'.format(l2_dist.mean(), l2_dist.std(), l2_dist.max(), l2_dist.min()))
					f.write('L1 norm - ave: {}\t std: {}\t max: {} min: {}\n'.format(l1_dist.mean(), l1_dist.std(), l1_dist.max(), l1_dist.min()))
					f.write('LInf norm - ave: {}\t std: {}\t max: {} min: {}\n'.format(inf_dist.mean(), inf_dist.std(), inf_dist.max(), inf_dist.min()))
					f.close()
				
				#print("L2 norm - ave: {}\t std: {}\t max: {} min: {}".format(l2_dist.mean(), l2_dist.std(), l2_dist.max(), l2_dist.min()))
				#print('L1 norm - ave: {}\t std: {}\t max: {} min: {}'.format(l1_dist.mean(), l1_dist.std(), l1_dist.max(), l1_dist.min()))
				#print("LInf norm - ave: {}\t std: {}\t max: {} min: {}".format(inf_dist.mean(), inf_dist.std(), inf_dist.max(), inf_dist.min()))
			for i in range(xs.shape[0]):
			#Print the first and 8th batches of images i.e. a batch of 2s and a batch of 7s
				enc_img = sess.run(encode_op, feed_dict={single_adv_x_op: adv_xs[i]}) #Only one image in a batch
				f = open(output_image_path + '/{}.png'.format(c), "wb+")
				f.write(enc_img)
				f.close()
				c += 1
	
		first_batch = False
		if adv_images is None:
			adv_images = np.array(adv_xs.reshape(adv_xs.shape[0], 28, 28))
		else:
			adv_images = np.append(adv_images, adv_xs.reshape(adv_xs.shape[0], 28, 28), 0)

	np.save(os.path.join(output_dir,'adv_predictions_{}'.format(attack_name)), all_adv_preds)
	acc = float(correct) / total
	time_required = time.time() - time_start
	print('Adv error: {:.3f}'.format((1 - acc) * 100))
	print('Time required to generate attack: {}'.format(time_required))
	
	if report_file is not None:
		f = open(report_file, "a+")
		f.write('{}	error: {}%\n'.format(attack_name, (1 - acc)*100))
		f.write('{}	time: {}s\n'.format(attack_name, time_required))
		f.close()
	
	np.save(output_dir + '/two_vs_seven_adv_{}'.format(attack_name), adv_images, allow_pickle=False)

def all_archs():
	#
	arch_list = [MNIST_arch_0, archs.MNIST_arch_1, archs.MNIST_arch_1b, archs.MNIST_arch_2, archs.MNIST_arch_3, archs.MNIST_arch_4, 
		archs.MNIST_arch_5, archs.MNIST_arch_6, archs.MNIST_arch_7, archs.MNIST_arch_8]
	for a in range(5, len(arch_list)):
		torch_model = train_model(data_dir=FLAGS.data_dir, model_arch=arch_list[a])
		torch.save(torch_model, FLAGS.model.format("arch_{}".format(a)))
		
		report_file = os.path.join('/home/squishymage/cnn_gp/archs', FLAGS.report)
		#import pdb; pdb.set_trace()
		if FLAGS.attack == 'fgsm':
			param_set = FGSM_Params(eps=FLAGS.eps,ord=FLAGS.ord)
		elif FLAGS.attack == 'cw_l2':
			param_set = CW_L2_Params(confidence=FLAGS.confidence, max_iterations=FLAGS.max_iterations, learning_rate=FLAGS.learning_rate, 
				binary_search_steps=FLAGS.binary_search_steps, initial_const=FLAGS.initial_const, batch_size=BATCH_SIZE)
		elif FLAGS.attack == 'ead':
			param_set = EAD_Params(beta=FLAGS.beta, confidence=FLAGS.confidence, max_iterations=FLAGS.max_iterations, learning_rate=FLAGS.learning_rate, 
				binary_search_steps=FLAGS.binary_search_steps, initial_const=FLAGS.initial_const, batch_size=BATCH_SIZE)
		elif FLAGS.attack == 'pgd':
			param_set = PGD_Params(eps=FLAGS.eps, eps_iter=FLAGS.eps_iter, ord=FLAGS.ord, nb_iter=FLAGS.nb_iter)
		else:
			print("ERROR - Param set not implemented")
			return


		generate_attack(FLAGS.attack, param_set, torch_model, FLAGS.data_dir, report_file=report_file, output_dir='/home/squishymage/cnn_gp/archs', attack_arch='arch_{}'.format(a))
	

def main(_=None):
	#from cleverhans_tutorials import check_installation
	#check_installation(__file__)
	
	if FLAGS.action == 'train':
		#Code for training and saving the CNN model that we're using for all the attacks
		torch_model = train_model(data_dir=FLAGS.data_dir)
		torch.save(torch_model, FLAGS.model)
		return

	#But now that we've saved it, we can just reload it every time:
	torch_model = torch.load(FLAGS.model, map_location=torch.device('cuda'))
	if FLAGS.action == 'generate':
		
		report_file = os.path.join(FLAGS.data_dir, FLAGS.report)
		#import pdb; pdb.set_trace()
		if FLAGS.attack == 'fgsm':
			param_set = FGSM_Params(eps=FLAGS.eps,ord=FLAGS.ord)
		elif FLAGS.attack == 'cw_l2':
			param_set = CW_L2_Params(confidence=FLAGS.confidence, max_iterations=FLAGS.max_iterations, learning_rate=FLAGS.learning_rate, 
				binary_search_steps=FLAGS.binary_search_steps, initial_const=FLAGS.initial_const, batch_size=BATCH_SIZE)
		elif FLAGS.attack == 'ead':
			param_set = EAD_Params(beta=FLAGS.beta, confidence=FLAGS.confidence, max_iterations=FLAGS.max_iterations, learning_rate=FLAGS.learning_rate, 
				binary_search_steps=FLAGS.binary_search_steps, initial_const=FLAGS.initial_const, batch_size=BATCH_SIZE)
		elif FLAGS.attack == 'pgd':
			param_set = PGD_Params(eps=FLAGS.eps, eps_iter=FLAGS.eps_iter, ord=FLAGS.ord, nb_iter=FLAGS.nb_iter)
		elif FLAGS.attack == 'spsa':
			param_set = SPSA_Params(eps=FLAGS.eps, nb_iter=FLAGS.nb_iter, spsa_iters=FLAGS.spsa_iters, learning_rate=FLAGS.learning_rate, delta=FLAGS.delta)
		else:
			print("ERROR - Param set not implemented")
			return


		generate_attack(FLAGS.attack, param_set, torch_model, FLAGS.data_dir, report_file=report_file, output_dir=FLAGS.data_dir)
		return
	
	if FLAGS.action == 'attack':
		print(FLAGS.attack_name)
		adv_images_file = os.path.join(FLAGS.data_dir, '{}/two_vs_seven_adv_{}.npy'.format(FLAGS.attack_name,FLAGS.attack_name))
		labels_file = os.path.join(FLAGS.data_dir, FLAGS.labels)
		report_file = os.path.join(FLAGS.output_path, FLAGS.report)
		transfer_attack(torch_model, attack_name=FLAGS.attack_name, adv_images_file=adv_images_file, labels_file=labels_file, report_file=report_file)
	

if __name__ == '__main__':
	flags.DEFINE_string('action', 'generate', 'Action to perform. Options are _train_ a new model, _generate_ a new attack for a given model and _attack_ to use an existing attack on the given model')
	#This directory should contain the MNIST zip files.
	flags.DEFINE_string('data_dir', '/media/etv21/Elements/CNN_Attacks/attacks', 'The directory containing the MNIST data files')
	#Inside the data_dir folder #Do we need this?
	flags.DEFINE_string('labels', 'two_vs_seven_labels.npy', 'File with labels for adversarial data. Should be inside the data_dir')
	#This is used to infer the location of the attack images. Not necessary to specify when generating attacks, then the name is inferred.
	flags.DEFINE_string('attack_name', 'cleverhans_FGSM_eps=0.1_norm_inf', 'Name of attack (eg. CNN_FGSM_eps=0.3_norm=inf). Used to infer location of adv data file.') 
	#This should be in the data_dir if we're generating an attack.
	#Otherwise, this should be in the adv_images's parent directory

	flags.DEFINE_string('output_path', '/media/etv21/Elements/CNN_Attacks/attacks', "Directory of report file")
	#Relative to data directory if generating, relative to the output directory if attacking. We really need to refactor this.
	flags.DEFINE_string('report', 'cnn_spsa_arch0.txt', "The CNNs accuracy will be appended to this file.")
	flags.DEFINE_string('model', '/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt', "The trained CNN")

	flags.DEFINE_string('attack', 'spsa', 'Identifier for the attack to run')

	flags.DEFINE_float('learning_rate', 5e-2, 'learning rate')
	flags.DEFINE_float('initial_const', 1.0, 'Initial value for the constant that determines the tradeoff b/w distortion and attack success')
	flags.DEFINE_float('confidence', 0, 'Confidence (Kappa)')
	flags.DEFINE_float('beta', 1e-2, 'Beta, tradeoff between L1 and L2 (EAD)')
	flags.DEFINE_integer('max_iterations', 1, 'Max iterations')
	flags.DEFINE_integer('binary_search_steps', 6, 'Max binary search steps for c')

	flags.DEFINE_integer('spsa_iters', 1, 'Number of SPSA iterations')
	flags.DEFINE_float('delta', 0.01, 'SPSA delta (step size for gradient approx)')

	flags.DEFINE_integer('nb_iter', 10, 'Number of iterations for PGD/optimization iters for spsa')
	flags.DEFINE_float('eps', 0.3, 'Max perturbation')

	flags.DEFINE_float('eps_iter', 0.05, 'Step perturbation')
	flags.DEFINE_float('ord', np.Inf, 'Order of norm')
	

	tf.app.run()
