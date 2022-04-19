from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from loss import dis_loss, gen_loss, grad_penalty
from model import Generator
from model import Discriminator

n_gpu = 1
device = torch.device('cuda:0')
lr = 0.002
beta1 = 0.5
batch_size = 32
image_size = 512
nc = 3
dim_latent = 512
num_epochs = 5
workers = 0
n_syn = 7
n_cov = 8
data_path = 'data'
save_path = './results/'
model_path = './model/'
data_size = 302652;
train_size = 0.8*data_size;
val_size = data_size - train_size;
num_epochs = 3
iteration = 0
start = 0
alpha = 0
gemma = 0.5

def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

def load_data(dataset, batch_size, image_size):
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),      
            transforms.RandomHorizontalFlip(),      
            transforms.ToTensor(),            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset.transform = transform
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    data_train, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=workers)
    return train_loader, test_loader

def save_img(tensor, i):
    grid = tensor[0]
    grid.clamp_(-1, 1).add_(1).div_(2)
    # Add 0.5 after normalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(f'{save_path}sample-iter{i}.png')

def save_model(models, names):
    for i in range(len(models)):
        model = models[i]
        name = names[i]
        torch.save(model.state_dict(), f'{model_path}{name}.pth')

dataset = dset.ImageFolder(root='data',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
data_train, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])
dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

    
def train(generator, discriminator, g_optim, d_optim, n_syn, dataloader, 
          iteration = 0, start = 0, alpha = 0, g_losses = [], d_losses = []):
    print("Starting Training\n")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            iteration += 1
            alpha = min(1, alpha + batch_size / train_size)
            
            real_img = data[0].to(device)
            discriminator.zero_grad()
            set_grad_flag(discriminator, True)
            set_grad_flag(generator, False)
            
            real_img.requires_grad = True
            if n_gpu > 1:
                real_pred = nn.parallel.data_parallel(discriminator, (real_img, alpha), range(n_gpu))
            else:
                real_pred = discriminator(real_img, alpha)
            
            latent = torch.randn((batch_size, dim_latent), device=device)
            noise = []
            for i in range(n_syn + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn((batch_size, 1, size, size), device=device))
            if n_gpu > 1:
                fake_img = nn.parallel.data_parallel(generator, (latent, noise, alpha), range(n_gpu))
                fake_pred = nn.parallel.data_parallel(discriminator, (fake_img, alpha), range(n_gpu))
            else:
                fake_img = generator(latent, noise, alpha).detach()
                fake_pred = discriminator(fake_img, alpha)
            d_loss = dis_loss(real_pred, fake_pred)
            if(gemma != None):
                gp = grad_penalty(discriminator, real_img, fake_img)
                d_loss = d_loss + gemma * gp
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            set_grad_flag(generator, True)
            set_grad_flag(discriminator, False)
            latent = torch.randn((batch_size, dim_latent), device=device)
            noise = []
            for i in range(n_syn + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn((batch_size, 1, size, size), device=device))
            if n_gpu > 1:
                fake_img = nn.parallel.data_parallel(generator, (latent, noise, alpha), range(n_gpu))
                fake_pred = nn.parallel.data_parallel(discriminator, (fake_img, alpha), range(n_gpu))
            else:
                fake_img = generator(latent, noise, alpha).detach()
                fake_pred = discriminator(fake_img, alpha)
            g_loss = gen_loss(fake_pred)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            if iteration % 20 == 0:
                save_img(fake_img.data.cpu(), iteration)
            if iteration % 1000 == 0:
                save_model([generator,discriminator,g_optim, d_optim, g_loss, d_loss], 
                           ['generator', 'discriminator', 'g_optim', 'd_optim', 'g_loss', 'd_loss'])
                save_model([(iteration, start, alpha, n_syn),(g_losses, d_losses)],['parameters','log'])
                print("model saved")
                
            
            
            
            
generator = Generator().to(device)
discriminator = Discriminator().to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
dataset = dset.ImageFolder(root=data_path)
train_loader, test_loader = load_data(dataset, batch_size, image_size)
g_losses = []
d_losses = []
train(generator, discriminator, g_optimizer, d_optimizer, n_syn, train_loader, iteration, start, alpha, g_losses, d_losses)
#Training loop

