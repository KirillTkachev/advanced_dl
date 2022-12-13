import itertools

import torch
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, device, learning_rate = 3e-4):
        super().__init__()
        self.device = device
        self.generator_A = Generator(in_channels, out_channels)
        self.discriminator_A = Discriminator(in_channels + out_channels)
        
        self.generator_B = Generator(out_channels, in_channels)
        self.discriminator_B = Discriminator(in_channels + out_channels)
        
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()
        
        self.gen_opt = torch.optim.Adam(itertools.chain(self.generator_A.parameters(), self.generator_B.parameters()), lr=learning_rate)
        self.disc_opt = torch.optim.Adam(itertools.chain(self.discriminator_A.parameters(), self.discriminator_B.parameters()), lr=learning_rate)
        
        self.generator_A.to(device)
        self.discriminator_A.to(device)
        self.generator_B.to(device)
        self.discriminator_B.to(device)
        
        
    def train(self, batch, alpha):
        self.generator_A.train()
        self.generator_B.train()
        self.discriminator_A.eval()
        self.discriminator_B.eval()
        
        real, condition = batch 
        real_a = condition.to(self.device)
        real_b = real.to(self.device)
        fake_b = self.generator_A(real_a)
        rec_a = self.generator_B(fake_b)
        fake_a = self.generator_B(real_b)
        rec_b = self.generator_A(fake_a)
        
        loss_A = self.l1(fake_b, real_b) * alpha
        loss_B = self.l1(fake_a, real_a) * alpha
        
        disc_A_logits = self.discriminator_A(fake_b, real_a).detach()
        loss_G_A = self.mse(disc_A_logits, torch.ones_like(disc_A_logits))
        
        disc_B_logits = self.discriminator_B(fake_a, real_b).detach()
        loss_G_B = self.mse(disc_B_logits, torch.ones_like(disc_B_logits))
        
        loss_cycle_A = self.l1(rec_a, real_a) * alpha
        loss_cycle_B = self.l1(rec_b, real_b) * alpha
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_A + loss_B
        
        self.gen_opt.zero_grad()
        loss_G.backward()
        self.gen_opt.step()
        
        self.generator_A.eval()
        self.generator_B.eval()
        self.discriminator_A.train()
        self.discriminator_B.train()
        
        fake_b = self.generator_A(real_a).detach()
        fake_a = self.generator_B(real_b).detach()
        
        fake_logits_A = self.discriminator_A(fake_b, real_a)
        real_logits_A = self.discriminator_A(real_b, real_a)
        fake_loss_A = self.mse(fake_logits_A, torch.zeros_like(fake_logits_A))
        real_loss_A = self.mse(real_logits_A, torch.ones_like(real_logits_A))

        discriminator_A_loss = (real_loss_A + fake_loss_A) / 2
        
        fake_logits_B = self.discriminator_B(fake_a, real_b)
        real_logits_B = self.discriminator_B(real_a, real_b)
        fake_loss_B = self.mse(fake_logits_B, torch.zeros_like(fake_logits_B))
        real_loss_B = self.mse(real_logits_B, torch.ones_like(real_logits_B))
        discriminator_B_loss = (real_loss_B + fake_loss_B) / 2
 
        self.disc_opt.zero_grad()
        discriminator_A_loss.backward()
        discriminator_B_loss.backward()   
        self.disc_opt.step()

        return loss_G.cpu().item(), discriminator_A_loss.cpu().item()
    
    def inference(self, batch):
        self.generator_A.eval()
        
        real, condition = batch
        real = real.to(self.device)
        condition = condition.to(self.device)
        
        with torch.no_grad():
            fake = self.generator_A(condition)
        
        return real, condition, fake