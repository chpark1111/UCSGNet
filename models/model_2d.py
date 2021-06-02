import torch
import torch.nn  as nn
import torch.nn.functional as F

FLOAT_EPS = torch.finfo(torch.float32).eps

class Encoder(nn.Module):
    def __init__(self, latent_sz=256):
        super(Encoder, self).__init__()

        self.latent_sz = latent_sz
        
        self.conv1 = nn.Conv2d(1, self.latent_sz // 8, 4, 2, padding = 2)
        self.conv2 = nn.Conv2d(self.latent_sz // 8, self.latent_sz // 4, 4, 2, padding = 2)
        self.conv3 = nn.Conv2d(self.latent_sz // 4, self.latent_sz // 2, 4, 2, padding = 2)
        self.conv4 = nn.Conv2d(self.latent_sz // 2, self.latent_sz, 4, 2, padding = 2)
        self.conv5 = nn.Conv2d(self.latent_sz, self.latent_sz, 4, 2)
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        '''
        Input: (batch_size, in_ch, 64, 64)
        Outputs: (batch_size, latent_sz)
        '''

        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))
        y = self.lrelu(self.conv3(y))
        y = self.lrelu(self.conv4(y))
        y = self.conv5(y)
        y = y.reshape(y.shape[0], -1)

        assert y.shape == torch.Size([y.shape[0], self.latent_sz])

        return y

class Decoder(nn.Module):
    def __init__(self, latent_sz=256):
        super(Decoder, self).__init__()

        self.latent_sz = latent_sz
        self.fc1 = nn.Linear(self.latent_sz, self.latent_sz * 2)
        self.fc2 = nn.Linear(self.latent_sz * 2, self.latent_sz * 4)
        self.fc3 = nn.Linear(self.latent_sz * 4, self.latent_sz * 8)

        self.lrelu = nn.LeakyReLU(0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        '''
        Input: (batch_size, latent_sz)
        Outputs: (batch_size, latent_sz * 8)
        '''
        y = self.lrelu(self.fc1(x))
        y = self.lrelu(self.fc2(y))
        y = self.lrelu(self.fc3(y))
        
        return y

class Converter(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.Tensor(1, 1, 1), requires_grad=True)
        nn.init.constant_(self.alpha, 1.)
        
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return (x / self.alpha.clamp(min = FLOAT_EPS)).clamp(self.min_val, self.max_val)

