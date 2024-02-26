import torch
import torch.nn as nn
import numpy as np
import random
from utils import AverageMeter, accuracy
import copy


class model_for_sharp():
    def __init__(self, model, dset_loaders, criterion, use_cuda=False):
        self.dataloader = dset_loaders
        self.criterion = criterion
        self.model = model
        self.functional = False
        self.use_cuda = use_cuda
        self.dim = 0   # get parameter dimension
        for p in self.model.parameters():
            self.dim += p.numel()

        self.train_loss, self.train_acc = self.compute_loss()
        #self.val_loss, self.val_acc = self.compute_loss(phase='val')

    def compute_loss(self, phase='train', ascent_stats= False):
        self.zero_grad()
        loss_mtr = AverageMeter()
        acc_mtr = AverageMeter()

        for inputs, targets in self.dataloader[phase]:
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.set_grad_enabled(ascent_stats):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                batch_acc = accuracy(outputs, targets, topk=(1,))[0]
                acc_mtr.update(batch_acc, inputs.size(0))

            loss_mtr.update(loss, inputs.shape[0])

            if ascent_stats:
                loss *= (-inputs.shape[0] / len(self.dataloader[phase].dataset))
                loss.backward()

        if ascent_stats:
            theta_star_params = []
            theta_star_grads = []
            for p in self.model.parameters():
                theta_star_params.append(copy.deepcopy(p))
                theta_star_grads.append(copy.deepcopy(p.grad.data))

            return theta_star_params, theta_star_grads, loss_mtr.avg.item(), acc_mtr.avg.item()
        else:
            return loss_mtr.avg.item(), acc_mtr.avg.item()

    def hvp(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        self.zero_grad()
        hessian_vec_prod = None
        phase = 'train'

        for inputs, targets in self.dataloader[phase]:
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_grad = torch.autograd.grad(
                grad_vec, self.model.parameters() , grad_outputs=vec, only_inputs=True)

            if hessian_vec_prod is not None:
                hessian_vec_prod += torch.cat([g.contiguous().view(-1) for g in grad_grad])
            else:
                hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])

            self.zero_grad()

        return hessian_vec_prod/len(self.dataloader[phase])
        #return grad_dict, grad_vec, grad_grad, hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()