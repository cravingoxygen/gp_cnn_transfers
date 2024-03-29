
class MNIST_arch_2(nn.Module):
  """ Basic MNIST model from github, unmodified
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_2, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(64*7*7, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


class MNIST_arch_3(nn.Module):
  """ Basic MNIST model from github, with fewer filters
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_3, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(32*7*7, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 32*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)
    
class MNIST_arch_4(nn.Module):
  """ Basic MNIST model from github, with more channels
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_4, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(128*7*7, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 128*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


class MNIST_arch_5(nn.Module):
  """ Basic MNIST model from github, smaller fully connected layer
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_5, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(64*7*7, 512)
    self.fc2 = nn.Linear(512, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)
    
class MNIST_arch_6(nn.Module):
  """ Basic MNIST model from github, with extra fully connected layer
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_6, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(64*7*7, 1024)
    self.fc2 = nn.Linear(1024, 1024)
    self.fc3 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = F.relu(self.fc2(x))
    x = F.dropout(x, training=self.training)
    x = self.fc3(x)
    return F.log_softmax(x)
    
class MNIST_arch_7(nn.Module):
  """ Basic MNIST model from github, with different filter sizes
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_7, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    # num_pixels - filter_size + 1 + 2*padding = layer_output_size
    self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 7, padding=3)
    # feature map size is 7*7 by pooling
    
    self.fc1 = nn.Linear(64*7*7, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64*7*7)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = F.relu(self.fc2(x))
    return F.log_softmax(x)
    
class MNIST_arch_8(nn.Module):
  """ Basic MNIST model from github, with extra filter layer
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_8, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.conv3 = nn.Conv2d(64, 128, 4, padding=2)
    # feature map is now 4*4 by pooling
    
    self.fc1 = nn.Linear(128*4*4, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    x = x.view(-1, 128*4*4)   # reshape Variable
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


class MNIST_arch_1(nn.Module):
  """ Basic MNIST model from github
  Modified to have one fc layer
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(MNIST_arch_1, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # Feature map is 14*14 after pooling
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.fc1 = nn.Linear(64 * 7 * 7, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64 * 7 * 7)  # reshape Variable
    x = self.fc1(x)
    return F.log_softmax(x, dim=-1)
    
