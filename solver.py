from networks import AdaINGen, Discriminator
# Variable是对tensor的封装，操作和tensor是一样的，但是每个Variable都有三个属性：
# Variable中的tensor本身.data，对应tensor的梯度.grad以及这个Variable是通过说明方式得到的.grad_fn
from torch.autograd import Variable
import torch
import os
import torch.nn.functional as F
import numpy as np
import time
import datetime
from torchvision.utils import save_image

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, face_loader, hyperparameters, opts):
        self.face_loader = face_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_dim = hyperparameters['gen']['style_dim']
        self.lr = hyperparameters['lr']
        self.label_dim = hyperparameters['label_dim']
        self.gen_var = hyperparameters['gen']
        self.dis_var =  hyperparameters['dis']
        self.max_iter = hyperparameters['max_iter']
        self.lambda_cls = hyperparameters['lambda_cls']
        self.lambda_gp = hyperparameters['lambda_gp']
        # fix the noise used in sampling
        batch_size = int(hyperparameters['batch_size'])
        # randn()返回一个张量，包含了从标准正态分布中抽取一组随机数，形状由可变参数sizes定义

        # Setup the optimizers
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.n_critic = hyperparameters['n_critic']
        self.lambda_rec = hyperparameters['lambda_rec']
        self.recon_s_w = hyperparameters['recon_s_w']
        self.recon_c_w = hyperparameters['recon_c_w']
        self.log_iter = hyperparameters['log_iter']
        self.image_save_iter = hyperparameters['image_save_iter']
        self.snapshot_save_iter = hyperparameters['snapshot_save_iter']
        self.output_path = opts.output_path
        self.model_save_dir = opts.model_save_dir
        self.step_size = hyperparameters['step_size']
        self.lr_update_step = hyperparameters['lr_update_step']
        self.result_dir = opts.result_dir
        self.num_style = opts.num_style

        self.build_model()


    def build_model(self):
        self.G = AdaINGen(self.label_dim, self.gen_var)
        self.D = Discriminator(self.label_dim, self.dis_var)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.D.to(self.device)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,# 每次 backward() 时，默认会把整个计算图free掉。
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=8):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        return F.cross_entropy(logit, target)

    def recon_criterion(self, input, target):
        # torch.mean()返回输入张量所有元素的均值；abs()输出张量的每个元素的绝对值
        return torch.mean(torch.abs(input - target))

    def train(self):
        data_loader = self.face_loader
        data_iter = iter(data_loader)
        # x_fixed表示图像像素值 c_org表示真实标签值
        x_fixed, c_org = next(data_iter)  # 得到一个batch的图片和标签
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.label_dim)
        g_lr = self.lr
        d_lr = self.lr

        start_iters = 0
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.max_iter):
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            # 给定参数n，返回一个从0 到n -1 的随机整数排列。
            rand_idx = torch.randperm(label_org.size(0))
            # 随机生成目标标签label_trg
            label_trg = label_org[rand_idx]
            c_org = self.label2onehot(label_org, self.label_dim)
            c_trg = self.label2onehot(label_trg, self.label_dim)
            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # dis_update
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)# d_loss_real最小，那么out_src最大==1（针对图像）
            d_loss_cls = self.classification_loss(out_cls, label_org)# 针对标签
            style = Variable(torch.randn(x_real.size(0), self.style_dim, 1, 1).to(self.device))
            # encode
            content_fake, _ = self.G.encode(x_real,c_trg)
            # decode
            x_fake = self.G.decode(content_fake, style, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)  # 假图像为0
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            # 最终d_loss_gp在0.9954~0.9956波动

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # gen_update
            if (i+1) % self.n_critic == 0:# 每更新5次判别器再更新一次生成器
                # encode
                content_real, style_real = self.G.encode(x_real, c_org)
                content_fake, style_fake = self.G.encode(x_real, c_trg)

                x_fake = self.G.decode(content_fake, style, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)  # 估计标签越接近目标标签损失越小
                x_recon = self.G.decode(content_real, style_real, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_recon))

                # encode again
                # content_recon, style_recon = self.G.encode(x_fake, c_trg)
                # reconstruction loss
                # self.loss_gen_recon_style = self.recon_criterion(style_recon, style)
                # self.loss_gen_recon_content = self.recon_criterion(content_recon, content_fake)

                # Backward and optimize.生成网络参数更新
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + \
                 self.lambda_cls * g_loss_cls
                 # + \
                 # self.recon_s_w * self.loss_gen_recon_style + \
                 # self.recon_c_w * self.loss_gen_recon_content

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                # loss['G/loss_style'] = self.loss_gen_recon_style.item()
                # loss['G/loss_content'] = self.loss_gen_recon_content.item()


            # Miscellaneous
            if (i + 1) % self.log_iter == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.max_iter)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            if (i + 1) % self.image_save_iter == 0:
                with torch.no_grad():
                    style1 = Variable(torch.randn(x_fixed.size(0), self.style_dim, 1, 1).to(self.device))
                    style2 = Variable(torch.randn(x_fixed.size(0), self.style_dim, 1, 1).to(self.device))
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        content_fake, style_fake = self.G.encode(x_fixed, c_fixed)
                        x_fake_list.append(self.G.decode(content_fake, style1, c_fixed))
                        x_fake_list.append(self.G.decode(content_fake, style2, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.output_path, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save network weights
            if (i + 1) % self.snapshot_save_iter == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.max_iter - self.step_size):
                g_lr -= (self.lr / float(self.step_size))
                d_lr -= (self.lr / float(self.step_size))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.max_iter)

        # Set data loader.
        data_loader = self.face_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, 8)

                # Translate images.
                x_fake_list = [x_real]
                style_rand = Variable(torch.randn(x_real.size(0), self.style_dim, 1, 1).cuda())

                for c_trg in c_trg_list:
                    content_fake, style_fake = self.G.encode(x_real, c_trg)
                    x_fake_list.append(self.G.decode(content_fake, style_rand, c_trg))

                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

















