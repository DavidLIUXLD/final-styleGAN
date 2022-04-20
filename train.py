from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.utils import make_grid
import torchvision.datasets as dset
import torchvision.transforms as transforms
from loss import dis_loss, gen_loss, grad_penalty
from model import Generator
from model import Discriminator

n_gpu = 1
device = torch.device('cuda')
lr = 0.001
beta1 = 0.5
batch_size = 8
image_size = 512
nc = 3
dim_latent = 256
n_in = 256
n_im = 256
num_epochs = 2
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
alpha = 0.1
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
    print(tensor.shape)
    for j in range(tensor.shape[0]):
        if(j % 2 == 0):
            grid = tensor[j]
            grid.clamp_(-1, 1).add_(1).div_(2)
            # Add 0.5 after normalizing to [0, 255] to round to nearest integer
            ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            img = Image.fromarray(ndarr)
            img.save(f'{save_path}sample-iter{i}_{j}.png')
    
def display_img(tensors):
    grid = make_grid(tensors, nrow = 2)
    img = torchvision.transforms.ToPILImage()(grid)
    img.show()

def save_model(models, names ,type):
    for i in range(len(models)):
        model = models[i]
        name = names[i]
        if type == 'model':
            torch.save(model.state_dict(), f'{model_path}{name}.pth')
        else:
            torch.save(model, f'{model_path}{name}.pth')
            
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
          iteration = 0, start = 0, alpha = 0, loss = None, g_losses = [], d_losses = []):
    print("Starting Training\n")
    true_label = 1.
    false_label = 0.
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            iteration += 1
            alpha = min(1, alpha + batch_size / train_size)
            
            #D
            real_img = data[0].to(device)
            discriminator.zero_grad()
            set_grad_flag(discriminator, True)
            set_grad_flag(generator, False)
            
            real_img.requires_grad = True
            if n_gpu > 1:
                real_pred = nn.parallel.data_parallel(discriminator, (real_img, alpha), range(n_gpu))
            else:
                real_pred = discriminator(real_img, alpha)
            if(loss != None):
                label = torch.full((batch_size,), true_label, dtype=torch.float, device=device)
                d_loss = loss(real_pred, label.view(1, -1).squeeze())
            else:
                d_loss = gen_loss(real_pred)
            d_loss.backward(retain_graph=True)
            
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
            if(loss != None):
                label = torch.fill_(label, false_label)
                d_loss = loss(fake_pred, label)
            else:
                d_loss = dis_loss(real_pred, fake_pred)
            if(gemma != None):
                gp = grad_penalty(discriminator, real_img, fake_img, batch_size, nc, device)
                d_loss = d_loss + gemma * gp
            d_loss.backward()
            d_optim.step()
            
            #G
            generator.zero_grad()
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
            if(loss != None):
                label = torch.fill_(label, true_label)
                g_loss = loss(fake_pred,label.view(1, -1).squeeze())
            else:
                g_loss = gen_loss(fake_pred)
            #g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            if iteration == 1:
                print("gen init pass" + str(g_loss))
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            print("iteration: " + str(iteration))
            if iteration % 100 == 0:
                loss_current = "g: " + str(g_loss) + " d: " + str(d_loss)
                print(loss_current) 
                img_list = fake_img[:4]
                display_img(img_list)
                save_img(fake_img.data.cpu(), iteration)
                with open('log.txt', 'w') as f:
                    f.write(loss_current)
            if iteration % 1000 == 0:
                save_model([generator,discriminator,g_optim, d_optim], 
                           ['generator', 'discriminator', 'g_optim', 'd_optim'], 'model')
                save_model([g_loss, d_loss], ['g_loss', 'd_loss]'], 'parae')
                save_model([(iteration, start, alpha, n_syn),(g_losses, d_losses)],['parameters','log'], 'para')
                print("model saved")
    return g_losses, d_losses

generator = Generator(latent_size=n_in, dlatent_size=dim_latent, im_ch=n_im).to(device)
discriminator = Discriminator(n_cov=n_cov, out_ch=n_in).to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
dataset = dset.ImageFolder(root=data_path)
train_loader, test_loader = load_data(dataset, batch_size, image_size)
g_losses = []
d_losses = []
Criterion = nn.BCEWithLogitsLoss()
generator.train()
discriminator.train()
train(generator, discriminator, g_optimizer, d_optimizer, n_syn, train_loader, iteration, start, alpha)