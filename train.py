import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from model import generator
from model import discriminator

device = torch.device('cuda')

netGen = generator().to(device)
'''if (device.type == 'cuda') and (ngpu > 1):
    netGen = nn.DataParallel(netG, list(range(ngpu)))'''

netDiscrim = discriminator().to(device)

lr = 0.000002
beta1 = 0.5
batch_size = 32
image_size = 512
nc = 3
dim_latent = 512
nz = 512
num_epochs = 5
workers = 0

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


'''
if (device.type == 'cuda') and (ngpu > 1):
    netDiscrim = nn.DataParallel(netD, list(range(ngpu)))
'''    
    

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, dim_latent, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerDiscrim = optim.Adam(netDiscrim.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerGen = optim.Adam(netGen.parameters(), lr=lr, betas=(beta1, 0.999))

#Training loop
img_list = []
Gen_losses = []
Discrim_losses = []
iters = 0

print("Starting Training\n")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        # (1) Update D network
        ## Train with all-real batch
        netDiscrim.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        print(b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netDiscrim(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errDiscrim_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errDiscrim_real.backward()
        Discrim_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        latent = torch.randn((b_size, dim_latent), device=device)
        # Generate fake image batch with G
        fake = netGen(latent)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netDiscrim(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errDiscrim_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errDiscrim_fake.backward()
        Discrim_Gen_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errDiscrim = errDiscrim_real + errD_fake
        # Update D
        optimizerDiscrim.step()

        # (2) Update G network:
        netGen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netDiscrim(fake).view(-1)
        # Calculate G's loss based on this output
        errGen = criterion(output, label)
        # Calculate gradients for G
        errGen.backward()
        Discrim_Gen_z2 = output.mean().item()
        # Update G
        optimizerGen.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errDiscrim.item(), errGen.item(), Discrim_x, Discrim_Gen_z1, Discrim_Gen_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netGen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
