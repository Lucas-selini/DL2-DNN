import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

### Inspired by https://github.com/lyeoni/pytorch-mnist-GAN

def load_mnist_gan(size,batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    idx = np.random.choice(len(train_dataset), size, replace=False)    
    train_dataset = torch.utils.data.Subset(train_dataset, idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    sample_img = train_dataset[0][0]
    mnist_dim = sample_img.size(1) * sample_img.size(2)

    return train_loader, mnist_dim

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim, g_hidden_dim, depth = 3):
        super(Generator, self).__init__()       
        self.g_input_dim = g_input_dim
        self.g_output_dim = g_output_dim
        g_hidden = g_hidden_dim
        self.depth = depth
        
        self.layers = nn.ModuleList([nn.Linear(self.g_input_dim, g_hidden_dim)])
        
        for _ in range(self.depth-1):
            self.layers.append(nn.Linear(g_hidden, g_hidden*2))
            g_hidden *= 2
        
        self.output_layer = nn.Linear(g_hidden, self.g_output_dim)
        
    def forward(self, x): 
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        return torch.tanh(self.output_layer(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim, d_hidden_dim, depth = 3):
        super(Discriminator, self).__init__()
        self.d_input_dim = d_input_dim
        self.d_hidden_dim = d_hidden_dim
        d_hidden = d_hidden_dim
        self.depth = depth
        
        self.layers = nn.ModuleList([nn.Linear(self.d_input_dim, d_hidden_dim)])
        
        for _ in range(self.depth-1):
            self.layers.append(nn.Linear(d_hidden, d_hidden//2))
            d_hidden //= 2
        
        self.output_layer = nn.Linear(d_hidden, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
            x = F.dropout(x, 0.3)
        
        return torch.sigmoid(self.output_layer(x))

class GAN:
    def __init__(self, g_input_dim, g_output_dim,g_hidden_dim, g_depth, d_input_dim,  d_hidden_dim, d_depth, lr=0.001, device='cpu'):
        self.g_input_dim = g_input_dim
        self.g_output_dim = g_output_dim
        self.d_input_dim = d_input_dim
        self.device = device
        
        self.generator = Generator(self.g_input_dim, self.g_output_dim, g_hidden_dim, g_depth ).to(self.device)
        self.discriminator = Discriminator(self.d_input_dim, d_hidden_dim, d_depth).to(self.device)
        self.loss_function = nn.BCELoss()

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
    
    def train_generator(self, bs):
        self.generator.zero_grad()
        noise = torch.randn(bs, self.g_input_dim).to(self.device)
        fake_data = self.generator(noise)
        decision = self.discriminator(fake_data)
        target = torch.ones(bs, 1).to(self.device)
        g_loss = self.loss_function(decision, target)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()
    
    def train_discriminator(self, real_data):
        bs = real_data.size(0)
        self.discriminator.zero_grad()

        real_target = torch.ones(bs, 1).to(self.device)
        fake_target = torch.zeros(bs, 1).to(self.device)

        real_data = real_data.view(bs, -1).to(self.device)
        real_decision = self.discriminator(real_data)
        real_loss = self.loss_function(real_decision, real_target)

        noise = torch.randn(bs, self.g_input_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_decision = self.discriminator(fake_data.detach())
        fake_loss = self.loss_function(fake_decision, fake_target)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            d_losses, g_losses = [], []
            for real_data, _ in train_loader:
                d_losses.append(self.train_discriminator(real_data))
                g_losses.append(self.train_generator(real_data.size(0)))

            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), epochs, torch.mean(torch.FloatTensor(d_losses)), torch.mean(torch.FloatTensor(g_losses))))

    def generate_images(self, num_images=15, device='cpu'):
        noise = torch.randn(num_images, self.g_input_dim).to(device)
        with torch.no_grad():
            generated_images = self.generator(noise)
        return generated_images

    def count_parameters(self):
        return sum(p.numel() for p in self.generator.parameters() if p.requires_grad) + sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

