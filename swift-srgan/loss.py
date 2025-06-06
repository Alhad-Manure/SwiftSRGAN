import torch
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision.models import mobilenet_v2

class GeneratorLoss(nn.Module):
    def __init__(self):
        #super(GeneratorLoss, self).__init__()
        super().__init__()
        mNetV2 = mobilenet_v2(pretrained=True) # Have replaced vgg with mobilenetV2
        loss_network = nn.Sequential(*list(mNetV2.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        #self.adv_loss = nn.BCELoss()

    def forward(self, out_labels, real_labels, out_images, target_images):
        
        # Relativistic Average GAN Loss for Generator
        # Generator wants: D(fake) - E[D(real)] to be high (fake should be more realistic than average real)
        # and D(real) - E[D(fake)] to be low (real should not be much more realistic than average fake)
        
        # Calculate averages
        avg_out_labels = torch.mean(out_labels)
        avg_real_labels = torch.mean(real_labels)

        # RaGAN generator loss
        # Loss for fake samples: we want D(fake) - E[D(real)] to be close to 1
        fake_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            out_labels - avg_real_labels, torch.ones_like(out_labels)
        ))
        
         # Loss for real samples: we want D(real) - E[D(fake)] to be close to 0
        real_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            real_labels - avg_out_labels, torch.zeros_like(real_labels)
        ))

        # Adversarial Loss
        adversarial_loss = (fake_loss + real_loss) / 2
        
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        

        # Check what range your images are in
        img_range = max(out_images.max().item(), target_images.max().item())
    
        # Use appropriate data_range
        if img_range > 10:  # Likely [0, 255]
            data_range = 255.0
        else:  # Likely [0, 1] or [-1, 1]
            data_range = img_range
    
        ssim_loss_val = 1 - ssim(out_images, target_images, data_range=data_range)


        #return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        #return image_loss + 0.002 * adversarial_loss + 0.02 * perception_loss + 5e-7 * tv_loss
        return image_loss + (0.2 * ssim_loss_val) + (0.0015 * adversarial_loss) + (0.015 * perception_loss) + (1e-6 * tv_loss)
        #return image_loss + (0.1 * ssim_loss_val) + (0.002 * adversarial_loss) +  (0.02 * perception_loss) +  (5e-7 * tv_loss)

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        #super(TVLoss, self).__init__()
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]