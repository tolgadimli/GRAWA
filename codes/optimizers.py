import torch
from torch.optim.optimizer import Optimizer, required

import torch

# inherited from: https://github.com/davda54/sam/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



# Experimental
class GRAWAOptimWrapper(Optimizer):
    def __init__(self, params, base_optimizer, grawa_layer_size, grawa_beta = 0.43, **kwargs):
        
        defaults = dict(grawa_beta = grawa_beta, grawa_layer_size=grawa_layer_size, **kwargs)
        super(GRAWAOptimWrapper, self).__init__(params, defaults)


        self.grawa_layer_size = grawa_layer_size

        # self.grawa_moving_grad_norm = torch.tensor(grawa_layer_size * [0], dtype=torch.float)
        # self.grawa_grad_norm_list = torch.tensor(grawa_layer_size * [0], dtype=torch.float)

        self.grawa_moving_grad_norm = torch.tensor(grawa_layer_size * [0], dtype=torch.float)
        self.grawa_grad_norm_list = torch.tensor(grawa_layer_size * [0], dtype=torch.float)
        self.grawa_grad_norm_est = torch.tensor(grawa_layer_size * [0], dtype=torch.float)

        self.beta = grawa_beta

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)


    def step(self, closure = None):

        
        loss = None
        if closure is not None:
            loss = closure()

        if not self.state:
            self.state["step"] = 1
        else: 
            self.state["step"] += 1

        j = 0
        wd = self.defaults['weight_decay']
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # print(torch.norm(p.grad).item())
                    self.grawa_grad_norm_list[j] = torch.norm(p.grad).item()
                    j = j + 1

        #print("Grad norm:", self.grawa_grad_norm_list)

        self.grawa_moving_grad_norm = self.beta * self.grawa_moving_grad_norm + (1 - self.beta) * self.grawa_grad_norm_list
        self.grawa_grad_norm_est = self.grawa_moving_grad_norm / (1 - self.beta ** self.state["step"])

        # print("Moving grad norm est:", self.grawa_grad_norm_est)
        # print("=================================================================================")
        self.base_optimizer.step()

        return loss
    
    def reset(self):
        self.grawa_moving_grad_norm = torch.tensor(self.grawa_layer_size * [0], dtype=torch.float)
        self.grawa_grad_norm_list = torch.tensor(self.grawa_layer_size * [0], dtype=torch.float)
        self.grawa_grad_norm_est = torch.tensor(self.grawa_layer_size * [0], dtype=torch.float)
        self.state["step"] = 0