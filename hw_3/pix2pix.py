import torch
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator


class Pix2Pix(nn.Module):
    def __init__(self, in_channels, out_channels, device, learning_rate = 3e-4):
        super().__init__()
        self.device = device
        self.generator = Generator(in_channels, out_channels)
        self.discriminator = Discriminator(in_channels + out_channels)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()

        self.generator.to(self.device)
        self.discriminator.to(self.device)
    
    def train(self, batch, alpha=100):
        self.generator.eval()
        self.discriminator.train()

        real, condition = batch
        real = real.to(self.device)
        condition = condition.to(self.device)
        
        fake_images = self.generator(condition).detach()
        fake_logits = self.discriminator(fake_images, condition)
        real_logits = self.discriminator(real, condition)
        fake_loss = self.mse(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.mse(real_logits, torch.ones_like(real_logits))
        
        discriminator_loss = (real_loss + fake_loss) / 2
        
        self.disc_opt.zero_grad()
        discriminator_loss.backward()
        self.disc_opt.step()

        self.generator.train()
        self.discriminator.eval()
        

        fake_images = self.generator(condition)
        disc_logits = self.discriminator(fake_images, condition)

        generator_loss = self.mse(disc_logits, torch.ones_like(disc_logits)) + alpha * self.l1(fake_images, real)
        
        self.gen_opt.zero_grad()
        generator_loss.backward()
        self.gen_opt.step()
        
        return generator_loss.cpu().item(), discriminator_loss.cpu().item()

    def inference(self, batch):
        self.generator.eval()
        
        real, condition = batch
        real = real.to(self.device)
        condition = condition.to(self.device)
        
        with torch.no_grad():
            fake = self.generator(condition)
        
        return real, condition, fake