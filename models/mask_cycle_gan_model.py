import torch
import itertools
from util.image_pool import ImagePool
from util.util import tensor2im
from .base_model import BaseModel
from . import networks
import re
import numpy as np
import cv2
import pandas as pd


class MaskCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']#, 'dis_Ars', 'dis_Asr', 'dis_Arr', 'dis_Bsr', 'dis_Brs', 'dis_Bss','comb_dis']
        self.loss_names_dis = ['dis_Ars', 'dis_Asr', 'dis_Arr', 'dis_Bsr', 'dis_Brs', 'dis_Bss','comb_dis']
        self.loss_names_mask = ['mask_Ars', 'mask_Asr', 'mask_Arr', 'mask_Bsr', 'mask_Brs', 'mask_Bss','comb_mask']
        # 'dis_Ars' - A: real to sim A to B, B: sim to real B to A
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.isMask = True
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # Load json
            self.real_df = pd.read_json('./datasets/data_Allsight/json_data/real_train_7_transformed.json').transpose()
            self.sim_df = pd.read_json('./datasets/data_Allsight/json_data/sim_train_7_transformed.json').transpose()
            # Load regrssion models
            self.sim_regressor = networks.define_regressor('./checkpoints/regression_models/white/real_7.pth', self.gpu_ids)
            self.real_regressor = networks.define_regressor('./checkpoints/regression_models/white/sim_7.pth', self.gpu_ids)

            self.img_dim = opt.crop_size
            
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionDistil = torch.nn.MSELoss()
            self.criterionMask = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        match1= re.search(r'(\d+).jpg$', input['A_paths'][0])
        match2= re.search(r'(\d+).jpg$', input['B_paths'][0])
        self.real_A_num = int(match1.group(1))
        self.real_B_num = int(match2.group(1))
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
        batch_x_ref = self.real_A # not included in single mode
        
        if self.isDistil:
            self.real_A_pose = self.real_regressor(self.real_A, batch_x_ref)
            self.fake_B_pose = self.sim_regressor(self.fake_B, batch_x_ref)
            self.rec_A_pose = self.real_regressor(self.rec_A, batch_x_ref)
            self.real_B_pose = self.sim_regressor(self.real_B, batch_x_ref)
            self.fake_A_pose = self.real_regressor(self.fake_A, batch_x_ref)
            self.rec_B_pose = self.sim_regressor(self.rec_B, batch_x_ref)
        
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients 
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # calculate Distil loss if active
        if self.isDistil: 
            self.loss_comb_dis = self.get_distil_loss_combined()
            self.loss_G +=  self.loss_comb_dis
        
        self.loss_comb_mask = self.get_mask_loss_combined()
        self.loss_G +=  self.loss_comb_mask
        
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def update_distil(self):
        """
        Update the value of lambda_C (a distillation hyperparameter) based on the current epoch.

        """
        # Start distillation if the current epoch matches the specified distillation epoch and distillation is enabled.
        if self.epoch_counter == self.opt.epoch_distil:
            self.start_distil()
            print("[INFO]  Distil loss activated")
            self.epoch_counter += 1
        else:
            self.epoch_counter += 1

        # Update lambda_C if distillation is enabled, based on the distillation policy and distil_epoch_count.
        if self.isDistil:
            self.lambda_C = self.distil_policy(self.distil_epoch_count)
            print(f"[INFO]  Distil loss lambda_C set to: {self.lambda_C}")
            self.distil_epoch_count += 1


    def get_distil_loss_combined(self):
        """
        Compute and return the combined distillation loss.

        Returns:
            float: The combined distillation loss computed by summing six different distillation losses.
        """
        if self.real_A_num > 4999:
            self.loss_dis_Ars = 0
            self.loss_dis_Asr = 0
            self.loss_dis_Arr = 0 
        else:     
            self.loss_dis_Ars = self.criterionDistil(self.real_A_pose, self.fake_B_pose) * self.lambda_C
            self.loss_dis_Asr = self.criterionDistil(self.fake_B_pose, self.rec_A_pose) * 0.5 * self.lambda_C
            self.loss_dis_Arr = self.criterionDistil(self.real_A_pose, self.rec_A_pose) * 0.5 * self.lambda_C
            
        if self.real_B_num > 4999:    
            self.loss_dis_Bsr = 0
            self.loss_dis_Brs = 0
            self.loss_dis_Bss = 0
        else:    
            self.loss_dis_Bsr = self.criterionDistil(self.real_B_pose, self.fake_A_pose) * self.lambda_C
            self.loss_dis_Brs = self.criterionDistil(self.fake_A_pose, self.rec_B_pose) * 0.5 * self.lambda_C
            self.loss_dis_Bss = self.criterionDistil(self.real_B_pose, self.rec_B_pose) * 0.5 * self.lambda_C
        
        return self.loss_dis_Ars + self.loss_dis_Asr + self.loss_dis_Arr + self.loss_dis_Bsr + self.loss_dis_Brs + self.loss_dis_Bss 
    
    def img_to_vis(self, vis, tens_imgs):
        
        tens_A = tens_imgs[0]*tens_imgs[1]
        tens_BB = tens_imgs[4]*tens_imgs[1]
        tens_A = tensor2im(tens_A)
        tens_A = tens_A.transpose([2, 0, 1])
        # Convert tensor images to NumPy arrays
        img_A = tensor2im(tens_imgs[0])
        img_A = img_A.transpose([2, 0, 1])
        # img_B = tensor2im(tens_imgs[2])
        # img_B = img_B.transpose([2, 0, 1])
        mask_A = tensor2im(tens_imgs[1])
        mask_A = mask_A.transpose([2, 0, 1])
        img_A_b = tensor2im(tens_BB)
        img_A_b = img_A_b.transpose([2, 0, 1])
        # mask_B = tensor2im(tens_imgs[3])
        # mask_B = mask_B.transpose([2, 0, 1])
        
        images = [img_A, mask_A, tens_A, img_A_b]#, img_B, mask_B]
        # Create a 2x2 grid
        grid_image = np.concatenate(images, axis=2)  # Combine horizontally
        # grid_image = np.concatenate(grid_image, axis=1)  # Combine vertically

        # Send the grid image data to Visdom for display
        vis.image(grid_image, win = 10)
        return
    
    def get_mask_loss_combined(self):
            """
            Compute and return the combined mask loss.

            Returns:
                float: The combined mask loss computed by summing six different mask losses.
            """
            lambda_D = self.opt.lambda_D
            if self.real_A_num > 4999: mask_A = 1
            else: 
                px_py_r_real = self.real_df['contact_px'][self.real_A_num] # from df + calib with resize of img
                px_py_r_real = (np.array(px_py_r_real)*self.img_dim)//480 
                mask_A = np.ones((self.img_dim, self.img_dim), dtype=np.uint8)
                mask_A = cv2.circle(mask_A, (int(px_py_r_real[0]), int(px_py_r_real[1])), int(px_py_r_real[2]), 0, thickness=-1)
                mask_A = torch.tensor(mask_A, dtype=torch.uint8).to(device=self.real_A.device)
                mask_A = mask_A.unsqueeze(0).expand(1, 3, -1, -1)
            if self.real_B_num > -1: mask_B = 1    
            else:
                px_py_r_sim = self.sim_df['contact_px'][self.real_B_num]
                px_py_r_sim = (np.array(px_py_r_sim)*self.img_dim)//480 
                mask_B = np.ones((self.img_dim, self.img_dim), dtype=np.uint8) 
                mask_B = cv2.circle(mask_B, (int(px_py_r_sim[0]), int(px_py_r_sim[1])), int(px_py_r_sim[2]), 0, thickness=-1)
                mask_B = torch.tensor(mask_B, dtype=torch.uint8).to(device=self.real_B.device) 
                mask_B = mask_B.unsqueeze(0).expand(1, 3, -1, -1)    
                
            self.loss_mask_Ars = self.criterionMask(self.real_A*mask_A, self.fake_B*mask_A) * lambda_D
            self.loss_mask_Asr = self.criterionMask(self.fake_B*mask_A, self.rec_A*mask_A) * 0.5 * lambda_D 
            self.loss_mask_Arr = self.criterionMask(self.real_A*mask_A, self.rec_A*mask_A) * 0.5 * lambda_D
            
            self.loss_mask_Bsr = self.criterionMask(self.real_B*mask_B, self.fake_A*mask_B) * lambda_D
            self.loss_mask_Brs = self.criterionMask(self.fake_A*mask_B, self.rec_B*mask_B) * 0.5 * lambda_D 
            self.loss_mask_Bss = self.criterionMask(self.real_B*mask_B, self.rec_B*mask_B) * 0.5 * lambda_D
            
            # for debug
            if self.real_A_num < 4999:
                self.img_to_vis(self.opt.vis, [self.real_A, mask_A, self.real_B, mask_B, self.fake_B])
            
            return self.loss_mask_Ars+self.loss_mask_Asr+self.loss_mask_Arr+self.loss_mask_Bsr+self.loss_mask_Brs+self.loss_mask_Bss 

    def distil_policy_rule(self, policy):
        """
        Define the distillation policy based on the given 'policy' parameter.

        Parameters:
            policy (str): The type of distillation policy. It can be either 'const' for a constant policy or 'linear' for a linear policy.

        Returns:
            function: The chosen distillation policy function that takes the current epoch as input and returns the value of lambda_C.
        """

        if policy == 'const':
            # Constant distillation policy that returns the initial value of lambda_C for all epochs.
            def const_rule(distil_epoch):
                return self.init_lambda_C
            rule = const_rule
            
        elif policy == 'linear':
            # Linear distillation policy that linearly changes the value of lambda_C over epochs.
            def linear_rule(distil_epoch):
                l_C = self.init_lambda_C + self.opt.distil_slope * distil_epoch
                return l_C
            rule = linear_rule
        
        return rule

