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


def fc_uniform_init(fc_layer):
  print('FC layer constant: {}'.format(fc_layer.weight.size(1)))
  stdev = 1./math.sqrt(fc_layer.weight.size(1))
  fc_layer.weight.data.uniform_(-stdev, stdev)
  if fc_layer.bias is not None:
    fc_layer.bias.data.uniform_(-stdev, stdev)
  return fc_layer

def conv2d_uniform_init(conv2d_layer):
  n = conv2d_layer.in_channels
  for k in conv2d_layer.kernel_size:
    n *= k
  print('conv layer constant: {}'.format(n))
  stdev = 1./math.sqrt(n)
  conv2d_layer.weight.data.uniform_(-stdev, stdev)
  if conv2d_layer.bias is not None:
    conv2d_layer.bias.data.uniform_(-stdev, stdev)
  return conv2d_layer

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


def test_pdf(num_samples=10000):
    test_loader = torch.utils.data.DataLoader(
      filter_dataset(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), 2, 7),
      batch_size=1, shuffle=False)
    all_preds = np.zeros(num_samples)
    
    for k in range(0, num_samples):
        torch_model = MNIST_arch_0()
        if torch.cuda.is_available():
            torch_model = torch_model.cuda()
        #Use the data loader to iterate, but break after the first pattern.
        for xs, ys in test_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds = torch_model(xs)
            preds_np = preds.data.cpu().numpy()
            all_preds[k] = preds_np[0][0]
            break
    #all_preds = np.exp(all_preds)
    plt.hist(all_preds, 100)
    import pdb; pdb.set_trace()
    plt.savefig("sanity_check_preds_1000_lnsoftmax.png".format(r), format='png', dpi=1200)
    plt.close()
    np.save('sanity_check_preds_1000_lnsoftmax',all_preds,)
    

def mnist_tutorial(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
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
  #np.save('/scratch/etv21/conv_gp_data/MNIST_data/expA/two_vs_seven_test', clean_images, allow_pickle=False)
  np.save('/scratch/etv21/conv_gp_data/MNIST_data/expA/two_vs_seven_labels', adv_labels, allow_pickle=False)
  
  return report
  
def parse_val(cnns_list, key):
  
  item_list = torch.zeros((cnns_list.size, cnns_list[0][key].shape[0]))
  
  #print('Forming list {}'.format(key))
  for n in range(0,cnns_list.size):
    item_list[n] = cnns_list[n][key]
  #print('Moving to gpu')
  #item_list = item_list.to(torch.device("cuda"))
  plt.hist(item_list.flatten(), 100)
  plt.savefig("{}_{}.png".format(key, cnns_list.size), format='png', dpi=100)
  plt.close()
  
  #print('Calculating mean')
  list_mean = item_list.mean().item()
  #print('Calculating variance')
  list_var = item_list.var().item()
  #print('Layer\tMean\tVariance')  
  print('{}:\t{}\t{}'.format(key, list_mean, list_var))
  return list_mean, list_var

def load_cnns():
  num_runs = 100
  cnns = np.empty(num_runs,dtype=object)
  
  for n in tqdm.trange(0,num_runs):
    with open("/scratch3/etv21/cnn_weights_biases/cnn_{}".format(n), 'rb') as f:
      cnns[n] = pickle.load(f)
      if n == 0:
        cnns[n]['cn1w'] = cnns[n]['cn1w'].to(torch.device("cuda"))
  
  print('Layer\tMean\tVariance')
  parse_val(cnns, 'cn1w')
  parse_val(cnns, 'cn1b')
  parse_val(cnns, 'cn2w')
  parse_val(cnns, 'cn2b')
  parse_val(cnns, 'fc1w')
  parse_val(cnns, 'fc1b')
  
def train_cnns():
  num_runs = 10000
  
  accuracies = np.zeros(num_runs)
  cnns = np.empty(num_runs,dtype=object)
  
  for n in tqdm.trange(0,num_runs):
    test_acc, torch_model = mnist_tutorial(nb_epochs=FLAGS.nb_epochs,batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate)
    accuracies[n] = test_acc
    cn1w, cn1b = torch_model.conv1.weight.detach().flatten().cpu(), torch_model.conv1.bias.detach().flatten().cpu()
    cn2w, cn2b = torch_model.conv2.weight.detach().flatten().cpu(), torch_model.conv2.bias.detach().flatten().cpu()
    fc1w, fc1b = torch_model.fc1.weight.detach().flatten().cpu(), torch_model.fc1.bias.detach().flatten().cpu()
    cnn_dict = {"cn1w": cn1w, "cn1b": cn1b, "cn2w":cn2w, "cn2b": cn2b, "fc1w": fc1w, "fc1b": fc1b}
    with open("/scratch3/etv21/cnn_weights_biases/cnn_{}".format(n), 'wb') as f:
      pickle.dump(cnn_dict, f)
    cnns[n] = cnn_dict
    
  np.save('/scratch3/etv21/cnn_weights_biases/accuracies_{}_runs'.format(num_runs), accuracies)
  

def main(_=None):
  #from cleverhans_tutorials import check_installation
  #check_installation(__file__)
  #train_cnns()
  #load_cnns()
  
  mnist_tutorial(nb_epochs=FLAGS.nb_epochs,batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate)
  
  #test_pdf()

if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()



'''
def filter_dataset(dataset, class_one, class_two):
    if dataset.train:
        images = dataset.train_data
        labels = dataset.train_labels
    else:
        images = dataset.test_data
        labels = dataset.test_labels

    class_one_mask = labels == class_one
    class_two_mask = labels == class_two
    extracted_images = torch.cat((images[class_one_mask], images[class_two_mask]))
    
    #Convert labels into 2-class labels:
    #one_hot_c1, one_hot_c2 = torch.tensor([1,0]), torch.tensor([0,1])
    #xtracted_labels = torch.cat((one_hot_c1.repeat(class_one_mask.sum(), 1), one_hot_c2.repeat(class_two_mask.sum(), 1)))
    extracted_labels = torch.cat((labels[class_one_mask]*0, labels[class_two_mask]/class_two))
    
    if dataset.train:
        dataset.train_data = extracted_images
        dataset.train_labels = extracted_labels
    else:
        dataset.test_data = extracted_images
        dataset.test_labels = extracted_labels
        
    return dataset
'''

def examine_weights_biases(torch_model):
  cn1_weights, cn1_bias = torch_model.conv1.weight.detach(), torch_model.conv1.bias.detach()
  cn2_weights, cn2_bias = torch_model.conv2.weight.detach(), torch_model.conv2.bias.detach()
  fc1_weights, fc1_bias = torch_model.fc1.weight.detach(), torch_model.fc1.bias.detach()
  all_weights, all_bias = torch.cat([cn1_weights.flatten(), cn2_weights.flatten(), fc1_weights.flatten()]), torch.cat([cn1_bias.flatten(), cn2_bias.flatten(), fc1_bias.flatten()])
  
  cn1_weights_mean, cn1_weights_var = torch.mean(cn1_weights), torch.var(cn1_weights)
  cn2_weights_mean, cn2_weights_var = torch.mean(cn2_weights), torch.var(cn2_weights)
  fc1_weights_mean, fc1_weights_var = torch.mean(fc1_weights), torch.var(fc1_weights)
  
  cn1_bias_mean, cn1_bias_var = torch.mean(cn1_bias), torch.var(cn1_bias)
  cn2_bias_mean, cn2_bias_var = torch.mean(cn2_bias), torch.var(cn2_bias)
  fc1_bias_mean, fc1_bias_var = torch.mean(fc1_bias), torch.var(fc1_bias)
  
  
  '''
  print("ConvLayer 1 weights: mean, variance = {}, {}".format(cn1_weights_mean, cn1_weights_var))
  print("ConvLayer 2 weights: mean, variance = {}, {}".format(cn2_weights_mean, cn2_weights_var))
  print("FC 1 weights: mean, variance = {}, {}".format(fc1_weights_mean, fc1_weights_var))

  print("ConvLayer 1 bias: mean, variance = {}, {}".format(cn1_bias_mean, cn1_bias_var))
  print("ConvLayer 2 bias: mean, variance = {}, {}".format(cn2_bias_mean, cn2_bias_var))
  print("FC 1 bias: mean, variance = {}, {}".format(fc1_bias_mean, fc1_bias_var))
  
  print("Overall weight mean and variance: {}, {}".format(torch.mean(all_weights), torch.var(all_weights)))
  print("Overall bias mean and variance: {}, {}".format(torch.mean(all_bias), torch.var(all_bias)))
  plt.hist(cn1_weights.flatten().cpu(), 100)
  plt.savefig("cn1_weights_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  plt.hist(cn1_bias.flatten().cpu(), 100)
  plt.savefig("cn1_bias_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  
  plt.hist(cn2_weights.flatten().cpu(), 100)
  plt.savefig("cn2_weights_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  plt.hist(cn2_bias.flatten().cpu(), 100)
  plt.savefig("cn2_bias_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  
  plt.hist(fc1_weights.flatten().cpu(), 100)
  plt.savefig("fc1_weights_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  plt.hist(fc1_bias.flatten().cpu(), 100)
  plt.savefig("fc1_bias_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  plt.hist(all_weights.cpu(), 100)
  plt.savefig("all_weights_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  plt.hist(all_bias.cpu(), 100)
  plt.savefig("all_bias_{}.png".format(r), format='png', dpi=1200)
  plt.close()
  '''
