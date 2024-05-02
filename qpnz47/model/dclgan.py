################################################################################################################################################################
# the code is mostly from https://github.com/JunlinHan/DCLGAN. It has been adapted to include multi-class loss, adaptive discriminator learning rate and logging
# the modifications made are as stated in my report submission
################################################################################################################################################################

import itertools
import torch
import numpy as np
import networks
from imagepool import ImagePool
from patchnce import PatchNCELoss


class DCLModel():
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    def __init__(self, 
                 nce_layers='4,8,12,16', 
                 input_nc=3, 
                 output_nc=3, 
                 ngf=64, 
                 ndf=64, 
                 netF='mlp_sample', 
                 netG='resnet_9blocks',
                 netD='basic',
                 normG='instance',
                 normD='instance',
                 n_layers_D=3,
                 no_dropout=True,
                 include_nonlocal=True,
                 init_type='xavier',
                 init_gain=0.02,
                 device='cpu',
                 pool_size=50,
                 num_patches=256,
                 lr=0.0001,
                 beta1=0.5,
                 beta2=0.999,
                 nce_idt=True,
                 lambda_GAN=1.0,
                 lambda_NCE=2.0,
                 lambda_IDT=1.0,
                 lambda_CLS=0.25,
                 multiclass=0,
                 start_decay_epoch = 25,
                 total_epochs = 100,
                 current_epoch = 0):
        
        super().__init__()

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.metrics = {'D_A' : [], 'G_A' : [], 'NCE1' : [], 'D_B' : [], 'G_B' : [], 'NCE2' : [], 'idt_B' : [], 'idt_A' : [], 
                       'D_A_real' : [], 'D_A_fake' : [], 'D_B_real' : [], 'D_B_fake' : []}
        self.metrics_averages = {'D_A' : [], 'G_A' : [], 'NCE1' : [], 'D_B' : [], 'G_B' : [], 'NCE2' : [], 'idt_B' : [], 'idt_A' : [], 
                       'D_A_real' : [], 'D_A_fake' : [], 'D_B_real' : [], 'D_B_fake' : []}

        self.optimizers = []
        self.discriminator_decay = 1.0
        self.device = device
        self.multiclass = multiclass
        self.use_multiclass = (multiclass > 0)

        if self.use_multiclass:
            self.metrics['D_A_cls'] = []
            self.metrics['D_B_cls'] = []
            self.metrics_averages['D_A_cls'] = []
            self.metrics_averages['D_B_cls'] = []

        self.start_decay_epoch = start_decay_epoch
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        
        self.nce_layers = [int(i) for i in nce_layers.split(',')]
        self.num_patches = num_patches
        self.nce_idt = nce_idt
        
        # loss coefficients
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_IDT = lambda_IDT
        self.lambda_CLS = lambda_CLS

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(input_nc, output_nc, ngf, netG, normG,
                                        not no_dropout, init_type, include_nonlocal, init_gain, self.device)
        self.netG_B = networks.define_G(input_nc, output_nc, ngf, netG, normG,
                                        not no_dropout, init_type, include_nonlocal, init_gain, self.device)
        self.netF1 = networks.define_F(input_nc, netF, normG,
                                       not no_dropout, init_type, init_gain, self.device)
        self.netF2 = networks.define_F(input_nc, netF, normG,
                                       not no_dropout, init_type, init_gain, self.device)

        self.netD_A = networks.define_D(output_nc, ndf, netD,
                                        n_layers_D, normD, init_type, init_gain, multiclass, self.device)
        self.netD_B = networks.define_D(output_nc, ndf, netD,
                                        n_layers_D, normD, init_type, init_gain, multiclass, self.device)
        
        self.fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
        # define loss functions
        
        self.criterionGAN = networks.GANLoss(use_multiclass=self.use_multiclass).to(self.device)
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss().to(self.device))

        self.criterionIdt = torch.nn.L1Loss().to(self.device)
        self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=lr, betas=(beta1, beta2))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        self.compute_G_loss().backward()  # calculate graidents for G
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
        self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake, target_classes=None):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real

        if self.use_multiclass:
            # predict both real/fake and class logits
            pred_real, pred_cls_real = netD(real)
            loss_D_real_ = self.criterionGAN(pred_real, True, predicted_classes=pred_cls_real, target_classes=target_classes)
            loss_D_real =  loss_D_real_['hinge']
            loss_D_multiclass_real = loss_D_real_['multiclass']
            # Fake
            pred_fake, pred_cls_fake = netD(fake.detach())
            loss_D_fake_ = self.criterionGAN(pred_fake, False, predicted_classes=pred_cls_fake, target_classes=target_classes)
            loss_D_fake =  loss_D_fake_['hinge']
            loss_D_multiclass_fake = loss_D_fake_['multiclass']
            
            # Combined loss and calculate gradients
            loss_D_hinge = (loss_D_real + loss_D_fake) * 0.5
            loss_D_multiclass = (loss_D_multiclass_real + loss_D_multiclass_fake) * 0.5
            loss_D = loss_D_hinge + loss_D_multiclass * self.lambda_CLS
            loss_D.backward()
            return loss_D, loss_D_multiclass, pred_real, pred_fake

        else:        
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D, pred_real, pred_fake

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # sampling image pool to stop discriminator from overfitting to newest samples
        
        if self.use_multiclass:
            fake_B, fake_B_classes = self.fake_B_pool.query(self.fake_B, [self.cls_A])
            self.loss_D_A, self.loss_D_A_multiclass, D_A_real, D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, fake_B, target_classes=fake_B_classes) * int(self.lambda_GAN)
            self.metrics['D_A_cls'].append(self.loss_D_A_multiclass.item())
            
        else:
            fake_B = self.fake_B_pool.query(self.fake_B) 
            self.loss_D_A, D_A_real, D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * int(self.lambda_GAN)
            
        self.metrics['D_A'].append(self.loss_D_A.item())
        self.metrics['D_A_real'].append(D_A_real.mean().item())
        self.metrics['D_A_fake'].append(D_A_fake.mean().item())

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # sampling image pool to stop discriminator from overfitting to newest samples

        if self.use_multiclass:
            fake_A, fake_A_classes = self.fake_A_pool.query(self.fake_A, [self.cls_B])
            self.loss_D_B, self.loss_D_B_multiclass, D_B_real, D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, fake_A, target_classes=fake_A_classes) * int(self.lambda_GAN)
            self.metrics['D_B_cls'].append(self.loss_D_B_multiclass.item())

        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B, D_B_real, D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * int(self.lambda_GAN)
        self.metrics['D_B'].append(self.loss_D_B.item())
        self.metrics['D_B_real'].append(D_B_real.mean().item())
        self.metrics['D_B_fake'].append(D_B_fake.mean().item())

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        if self.use_multiclass:
            # predict both real/fake and class logits
            pred_fakeB, pred_cls_fakeB = self.netD_A(fakeB)
            pred_fakeA, pred_cls_fakeA = self.netD_B(fakeA)
            loss_G_A_ = self.criterionGAN(pred_fakeB, True, predicted_classes=pred_cls_fakeB, target_classes=self.cls_A)
            self.loss_G_A = (loss_G_A_['hinge'] + loss_G_A_['multiclass']).mean() * self.lambda_CLS
            self.loss_G_A = self.loss_G_A * self.lambda_GAN
            
            loss_G_B_ = self.criterionGAN(pred_fakeA, True, predicted_classes=pred_cls_fakeA, target_classes=self.cls_B)
            self.loss_G_B = (loss_G_B_['hinge'] + loss_G_B_['multiclass']).mean() * self.lambda_CLS
            self.loss_G_B = self.loss_G_B * self.lambda_GAN

        else:
            # just predict logits
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.lambda_GAN
    
        
        self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.lambda_NCE
        self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.lambda_NCE

        # L1 IDENTICAL Loss
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_IDT
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_IDT

        self.metrics['G_A'].append(self.loss_G_A.item())
        self.metrics['G_B'].append(self.loss_G_B.item())

        self.metrics['NCE1'].append(self.loss_NCE1.item())
        self.metrics['NCE2'].append(self.loss_NCE2.item())

        self.metrics['idt_A'].append(self.loss_idt_A.item())
        self.metrics['idt_B'].append(self.loss_idt_B.item())
        
        loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):
        n_layers = len(self.nce_layers)
        # first, get the features from the fake B (src) from the network which encodes B->A
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        # now, get the features from the real A (src) from the network which encodes A->B
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        # sample IDs from the real A
        feat_k_pool, sample_ids = self.netF1(feat_k, self.num_patches, None)
        # get the features relating to those sample IDs in the generated image
        feat_q_pool, _ = self.netF2(feat_q, self.num_patches, sample_ids)
        total_nce_loss = 0.0
        
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            # now, compute feature-by-feature loss using cosine similarity and cross entropy as in patchnce.py
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        n_layers = len(self.nce_layers)
        # first, get the features from the fake A (src) from the network which encodes A->B
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        # now, get the features from the real B (src) from the network which encodes B->A
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        # sample IDs from the real B
        feat_k_pool, sample_ids = self.netF2(feat_k, self.num_patches, None)
        # get the features relating to those sample IDs in the generated image
        feat_q_pool, _ = self.netF1(feat_q, self.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            # now, compute feature-by-feature loss using cosine similarity and cross entropy as in patchnce.py
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        # set class labels if we are using the multiclass inputs
        if self.use_multiclass:
            self.real_A, self.cls_A = input['A'][0].to(self.device), input['A'][1].to(self.device)
            self.real_B, self.cls_B = input['B'][0].to(self.device), input['B'][1].to(self.device)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)
            
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self, starting_lr, decay_discriminator_factor):

        if (self.metrics_averages["D_B_real"][-1] + self.metrics_averages["D_A_real"][-1])/2 - (self.metrics_averages["D_B_fake"][-1] + self.metrics_averages["D_A_fake"][-1])/2  > 0.8:
            self.discriminator_decay = decay_discriminator_factor * self.discriminator_decay
            
        for optim in self.optimizers: 
            if optim == self.optimizer_D: 
                networks.adjust_learning_rate(optim, starting_lr, self.current_epoch, self.start_decay_epoch, self.total_epochs, decay_param=self.discriminator_decay)
            else:
                networks.adjust_learning_rate(optim, starting_lr, self.current_epoch, self.start_decay_epoch, self.total_epochs)
            
        

        self.current_epoch += 1

    def update_loss_dictionaries(self):
        for key in self.metrics.keys():
            self.metrics_averages[key].append(np.mean(self.metrics[key]))
            self.metrics[key] = []
        return self.metrics_averages

    def save_models(self, destination_directory):
        print("SAVING MODELS")
        torch.save({
            'netG_A': self.netG_A.state_dict(),
            'netG_B': self.netG_B.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'netF1':  self.netF1.state_dict(),
            'netF2':  self.netF2.state_dict(),
        }, f'{destination_directory}/dclgan_params.pth')

    def save_generator(self, destination_directory, direction):
        if direction == 'AtoB':
            torch.save(self.netG_A, f'{destination_directory}/best_netG_A')
        elif direction == 'BtoA':
            torch.save(self.netG_A, f'{destination_directory}/best_netG_B')

    def load_models(self, source_directory):
        
        checkpoint = torch.load(f'{source_directory}/dclgan_params.pth')
        self.netG_A.load_state_dict(checkpoint['netG_A'])
        self.netG_B.load_state_dict(checkpoint['netG_B'])

        self.netD_A.load_state_dict(checkpoint['netD_A'])
        self.netD_B.load_state_dict(checkpoint['netD_B'])

        self.netF1.load_state_dict(checkpoint['netF1'])
        self.netF2.load_state_dict(checkpoint['netF2'])