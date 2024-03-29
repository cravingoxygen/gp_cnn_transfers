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
import tensorflow as tf
from tensorflow.python.platform import flags
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

import matplotlib.pyplot as plt
from PIL import Image 
import pickle
import tqdm
import math

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
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

def mnist_sevens_vs_twos(data_path, noisy=True, float_type=np.float64): 
    from tensorflow.examples.tutorials.mnist import input_data
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    ds = input_data.read_data_sets(data_path, one_hot=True)
    
    train_images, train_labels = extract_images_and_labels(2, 7, ds.train.images, ds.train.labels)
    test_images, test_labels = extract_images_and_labels(2, 7, ds.test.images, ds.test.labels)
    #validation_images, validation_labels = extract_images_and_labels(2, 7, ds.validation.images, ds.validation.labels)

    if noisy:
        train_noise = np.random.normal(0, 0.0001, train_images.shape)
        test_noise = np.random.normal(0, 0.0001, test_images.shape)
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
        self.images = images.reshape(-1, 1, 28, 28).astype(np.float32)
        self.labels = labels.astype(np.long)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class Adversarial_MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path='/scratch/etv21/conv_gp_data/MNIST_data/reverse/gp_adversarial_examples_eps=0.3_norm_inf.npy', label_path='/scratch/etv21/conv_gp_data/MNIST_data/eps=0.3_arch_0/two_vs_seven_labels.npy'):
        images = np.load(image_path).reshape(-1, 28,28)
        labels = np.load(label_path)
        
        assert images.shape[0] == labels.shape[0]
        self.images = np.expand_dims(images, 1).astype(np.float32)
        self.labels = labels.astype(np.float32)

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
  for k in conv2d_layer.kernel_size:
    n *= k
  print('conv layer constant: {}'.format(n))
  stdev = 1./math.sqrt(n)
  conv2d_layer.weight.data.normal_(0, stdev)
  if conv2d_layer.bias is not None:
    conv2d_layer.bias.data.normal_(0, stdev)
  return conv2d_layer

def mnist(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   train_end=-1, test_end=-1, learning_rate=LEARNING_RATE):
  """
  MNIST cleverhans tutorial
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :return: an AccuracyReport object
  """
  # Train a pytorch MNIST model
  torch_model = MNIST_arch_0()
  if torch.cuda.is_available():
    torch_model = torch_model.cuda()
  report = AccuracyReport()
  data_dir = '/scratch/etv21/conv_gp_data/MNIST_data/expA'
  training_dataset, test_dataset = mnist_sevens_vs_twos(data_dir ,noisy=True)
  train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
  #adversarial_loader = torch.utils.data.DataLoader(Adversarial_MNIST_Dataset(), batch_size=batch_size)

  # Train our model
  optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)
  train_loss = []

  total = 0
  correct = 0
  step = 0
  for _epoch in range(nb_epochs):
    for xs, ys in train_loader:
      xs, ys = Variable(xs), Variable(ys)
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      optimizer.zero_grad()
      preds = torch_model(xs)
      loss = F.nll_loss(preds, ys)
      loss.backward()  # calc gradients
      train_loss.append(loss.data.item())
      optimizer.step()  # update gradients

      preds_np = preds.data.cpu().numpy()
      correct += (np.argmax(preds_np, axis=1) == ys.cpu().numpy()).sum()
      total += len(xs)
      step += 1
      
      if total % 200 == 0:
        acc = float(correct) / total
        print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
        total = 0
        correct = 0
      

  #examine_weights_biases(torch_model)
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
  report.clean_train_clean_eval = acc
  print('[%s] Clean accuracy: %.2f%%' % (step, acc * 100))
  
  '''
  For transfer from GP examples to CNN:
  total = 0
  correct = 0
  
  #import pdb; pdb.set_trace()
  c = 0
  for xs, ys in adversarial_loader:
    xs, ys = Variable(xs), Variable(ys)
    if torch.cuda.is_available():
      xs, ys = xs.cuda(), ys.cuda()
    
    preds = torch_model(xs)
    preds_np = preds.data.cpu().numpy()

    correct += (np.argmax(preds_np, axis=1) == ys.cpu().numpy()).sum()
    total += len(xs)

  acc = float(correct) / total
  print('[%s] Adversarial accuracy: %.2f%%' % (step, acc * 100))

 '''
  
  # We use tf for evaluation on adversarial data
  sess = tf.Session()
  x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

  # Convert pytorch model to a tf_model and wrap it in cleverhans
  tf_model_fn = convert_pytorch_model_to_tf(torch_model)
  cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

  # Create an FGSM attack
  fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
  epsilon = 10
  norm = 2
  fgsm_params = {'eps': epsilon,
                 'clip_min': 0.,
                 'clip_max': 1.,
                 'ord': norm}
                 
  attack_name = 'CNN_FGSM_eps={}_norm={}'.format(epsilon, norm)
  attack_dir = os.path.join(data_dir, attack_name)
  if not os.path.exists(attack_dir):
    os.makedirs(attack_dir)
    print("Directory " , attack_dir ,  " Created ")
    

  adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
  adv_preds_op = tf_model_fn(adv_x_op)
  
  # Run an evaluation of our model against fgsm
  total = 0
  correct = 0
  
  all_adv_preds = np.array(0)
  for xs, ys in test_loader:
    adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
    all_adv_preds = np.append(all_adv_preds, adv_preds)
    correct += (np.argmax(adv_preds, axis=1) == ys.cpu().numpy()).sum()
    total += len(xs)

  np.save('adv_predictions', all_adv_preds)
  acc = float(correct) / total
  print('Adv accuracy: {:.3f}'.format(acc * 100))
  report.clean_train_adv_eval = acc
  
  
  single_adv_x_op = tf.placeholder(tf.float32, shape=(1,28,28))
  encode_op = tf.image.encode_png(tf.reshape(tf.cast(single_adv_x_op*255, tf.uint8), (28, 28, 1)))

  adv_images, clean_images, adv_labels = None, None, None

  #Print the first and 8th batches of images i.e. a batch of 2s and a batch of 7s
  b = 0
  for xs, ys in test_loader:
    adv_xs = sess.run(adv_x_op, feed_dict={x_op: xs})
    
    if b == 0 or b == 10:
      c = b*batch_size
      for i in range(0,adv_xs.shape[0]):
          enc_img = sess.run(encode_op, feed_dict={single_adv_x_op: adv_xs[i]})
          
          f = open('/scratch/etv21/conv_gp_data/MNIST_data/expA/{}/{}.png'.format(attack_name, c), "wb+")
          f.write(enc_img)
          f.close()
          c += 1
  
    if adv_images is None:
        adv_images = np.array(adv_xs.reshape(adv_xs.shape[0], 28, 28))
        clean_images = np.array(xs.reshape(xs.shape[0], 28, 28))
        adv_labels = np.array(ys)
    else:
        adv_images = np.append(adv_images, adv_xs.reshape(adv_xs.shape[0], 28, 28), 0)
        clean_images = np.append(clean_images, xs.reshape(xs.shape[0], 28, 28), 0)
        adv_labels = np.append(adv_labels, ys, 0)
    b += 1
  
  np.save('/scratch/etv21/conv_gp_data/MNIST_data/expA/two_vs_seven_adv_{}'.format(attack_name), adv_images, allow_pickle=False)
  np.save('/scratch/etv21/conv_gp_data/MNIST_data/expA/two_vs_seven_labels', adv_labels, allow_pickle=False)
  
  return report
  

def main(_=None):
  #from cleverhans_tutorials import check_installation
  #check_installation(__file__)
  #train_cnns()
  #load_cnns()
  
  mnist(nb_epochs=FLAGS.nb_epochs,batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate)
  
  #test_pdf()

if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS, 'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate for training')
  
  flags.DEFINE_bool('generate_adversarial', False, 'Whether to generate a new attack, or use the existing, specified file')
  flags.DEFINE_string('adv_data', LEARNING_RATE, 'File to use as adversarial data')

  tf.app.run()
