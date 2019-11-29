import numpy as np
import torch
from torch import nn
from tensorflow.python.platform import flags
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import tensorflow as tf

#Use cnn because it's already wrapped in the right structures to use for data loaders
from cnn import mnist_sevens_vs_twos

f = flags.FLAGS

num_epochs = 200
batch_size = 128
learning_rate = 1e-3

class Autoencoder(nn.Module):
    def __init(self):
        super(Autoencoder, self).__init()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, e):
        return self.decoder(e)

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

def main(_):
    FLAGS = flags.FLAGS

    #Train the Autoencoder

    model = Autoencoder().cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_dataset, test_dataset = mnist_sevens_vs_twos(FLAGS.data_path, noisy=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if not os.path.exists('{}/mlp_img'.format(FLAGS.output_dir)):
        os.mkdir('{}/mlp_img'.format(FLAGS.output_dir))

    for epoch in range(num_epochs):
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            MSE_loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
        if epoch % 10 == 0:
            x = to_img(img.cpu().data)
            x_hat = to_img(output.cpu().data)
            save_image(x, '{}/mlp_img/x_{}.png'.format(FLAGS.output_dir, epoch))
            save_image(x_hat, '{}/mlp_img/x_hat_{}.png'.format(FLAGS.output_dir, epoch))

    torch.save(model.state_dict(), '{}/sim_autoencoder.pth'.format(FLAGS.model_path))
    


    #Load encoder so we can use it:
    model = Autoencoder()
    model.load_state_dict(torch.load('{}/sim_autoencoder.pth'.format(FLAGS.model_path)))

    #Now we need to generate a 2-D version of the 2vs7 MNIST dataset, so that we can use that for training.
    set_names = ["train", "test"]
    loaders = [train_loader, test_loader]

    for i, loader in loaders:
        encoded_dataset = None
        for data in loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            
            enc_img = model.encode(img)
            if encoded_dataset is None:
                encoded_dataset = np.array(enc_img.reshape(enc_img.shape[0], 28, 28))
            else:
                encoded_dataset = np.append(encoded_dataset, enc_img.reshape(enc_img.shape[0], 28, 28), 0)
            
            """ print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
                .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
            if epoch % 10 == 0:
                x = to_img(img.cpu().data)
                x_hat = to_img(output.cpu().data)
                save_image(x, '{}/mlp_img/x_{}.png'.format(FLAGS.output_dir, epoch))
                save_image(x_hat, '{}/mlp_img/x_hat_{}.png'.format(FLAGS.output_dir, epoch)) """
        
        np.save(FLAGS.output_dir + '/two_vs_seven_enc_{}'.format(set_names[i]), encoded_dataset, allow_pickle=False)

    import pdb; pdb.set_trace()




if __name__ == '__main__':
    flags.DEFINE_integer('seed', 20, 'The seed to use')

    flags.DEFINE_string('data_path', '/scratch/etv21/conv_gp_data/MNIST_data/expA2', "Path to the compressed dataset")
    flags.DEFINE_string('model_path', '/scratch/etv21/boundary_dists', "Location where autoencoder will be saved/loaded from")
    flags.DEFINE_string('output_dir', '/scratch/etv21/boundary_dists', "Location where generated files will be placed (graphs, kernels, etc)")
    tf.app.run()