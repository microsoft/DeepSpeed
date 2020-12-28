import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from time import time

from gan_model import Generator, Discriminator, weights_init
from utils import get_argument_parser, set_seed, create_folder

import deepspeed

def get_dataset(args):
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.dataroot is None and str(args.dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % args.dataset)
    if args.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=args.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        nc=3
    elif args.dataset == 'lsun':
        classes = [ c + '_train' for c in args.classes.split(',')]
        dataset = dset.LSUN(root=args.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.CenterCrop(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif args.dataset == 'mnist':
            dataset = dset.MNIST(root=args.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
            nc=1

    elif args.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                                transform=transforms.ToTensor())
        nc=3

    elif args.dataset == 'celeba':
        dataset = dset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.CenterCrop(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        nc = 3

    assert dataset
    return dataset, nc

def train(args):
    writer = SummaryWriter(log_dir=args.tensorboard_path)
    create_folder(args.outf)
    set_seed(args.manualSeed)
    cudnn.benchmark = True
    dataset, nc = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank) #torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = 0
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    netG = Generator(ngpu, ngf, nc, nz).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))

    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    model_engineD, optimizerD, _, _ = deepspeed.initialize(args=args, model=netD, model_parameters=netD.parameters(), optimizer=optimizerD)
    model_engineG, optimizerG, _, _ = deepspeed.initialize(args=args, model=netG, model_parameters=netG.parameters(), optimizer=optimizerG)

    torch.cuda.synchronize()
    start = time()
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size,), real_label, dtype=real.dtype, device=device)
            output = netD(real)
            errD_real = criterion(output, label)
            model_engineD.backward(errD_real)
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            model_engineD.backward(errD_fake)
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            #optimizerD.step() # alternative (equivalent) step
            model_engineD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            model_engineG.backward(errG)
            D_G_z2 = output.mean().item()
            #optimizerG.step() # alternative (equivalent) step
            model_engineG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Loss_D", errD.item(), epoch*len(dataloader)+i)
            writer.add_scalar("Loss_G", errG.item(), epoch*len(dataloader)+i)
            if i % 100 == 0:
                vutils.save_image(real,
                        '%s/real_samples.png' % args.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                        normalize=True)

        # do checkpointing
        #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
    torch.cuda.synchronize()
    stop = time()
    print(f"total wall clock time for {args.epochs} epochs is {stop-start} secs")

def main():
    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
