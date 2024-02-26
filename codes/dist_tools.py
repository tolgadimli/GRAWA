import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

def get_model_layer_num( model ):
    """return number of layer components in the model"""
    layer_size = 0
    for _,_ in model.named_parameters():
        #print(k)
        layer_size += 1
    return layer_size

def get_grad_norm_tensor( optimizer, model, dataloader, device, level = 'layer' ):
    """return accumulated gradient in each layer component as a dict"""

    # wd = optimizer.param_groups[0]['weight_decay']

    assert level == 'layer' or level == 'model', 'Level can be layer or model...'
    total_grad_dict = {k:0 for k,_ in model.named_parameters()} 
    #print(total_grad_dict.keys())
    for  (data, target) in dataloader:
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()

        #grad_dict = {k:v.grad for k,v in model.named_parameters()}
        for k, v in model.named_parameters():
            total_grad_dict[k] += v.grad

    ls = []
    for k, v in total_grad_dict.items():
        #all_grad_norms[k] = torch.norm( v).item() 
        ls.append(torch.norm(v).item() ) 

    if level == 'layer':
        pass
    elif level == 'model':
        ls = [sum(ls)]

    grad_norm_tensor = torch.Tensor(ls)
    return grad_norm_tensor
    



def weight_model_params(model, flat_tensor, layer_comp_weights):
    '''assign the flat tensor to model's parameters using a weighted scheme where weights are specified in layer_comp_weights'''

    cnt = 0
    for parameter in model.parameters():
        cnt += 1
    assert len(layer_comp_weights) == cnt, 'weight coefficient should be in the range of [0,1]'

    #current_index = 0
    for w, parameter in zip(layer_comp_weights, model.parameters()):
        # numel = parameter.data.numel()
        # size = parameter.data.size()

        new_content = torch.mul( parameter.data, w )
        parameter.data.copy_( new_content )
        #current_index += numel


def copy_weighted_model_params(flat_tensor, model, layer_comp_weights, level):
    '''copy model parameters into the flat tensor'''
    if level == 'layer':
        current_index = 0 
        ind = 0
        for parameter in model.parameters():
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].data.copy_( torch.mul(layer_comp_weights[ind], parameter.data.view(-1)) )
            current_index += numel
            ind += 1

    elif level == 'model':
        current_index = 0 
        ind = 0
        for parameter in model.parameters():
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].data.copy_( torch.mul(layer_comp_weights[0], parameter.data.view(-1)) )
            current_index += numel
            ind += 1     


def set_global_rank(args, cur_worker):
    '''return global rank base on the index of current group and the index of current work inside the group'''
    global_rank = cur_worker + args.cur_group* args.num_gpus
    return global_rank

def dist_print(args, text2print):
    '''print out information'''
    if dist.get_rank() % args.num_gpus == 0:
        print(text2print)

def dist_dumb_barrier(args, device):
    '''alternative version of barrier to replace PyTorch's impletation of barrier'''
    dumb_tensor = torch.FloatTensor([1]).to(device)
    dist.all_reduce_multigpu([dumb_tensor], dist.ReduceOp.SUM)
    assert dumb_tensor.item() == args.world_size, 'NCCL was not intialized properly'

def ravel_model_params(model, is_grad, device):
    '''squash model parameters or gradients into a flat tensor (https://github.com/ucla-labx/distbelief)'''
    numel = 0
    for parameter in model.parameters():
        numel += parameter.data.numel()
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model.parameters():
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def ravel_model_buffers(model, is_grad, device):
    '''squash model buffers or gradients into a flat tensor'''
    numel = 0
    for parameter in model.buffers():
        numel += parameter.data.numel()
    if numel == 0:
        return None
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model.buffers():
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def mix_model_params(model, flat_tensor, tensor_weight=1):
    '''squash model parameters or gradients into the flat tensor'''
    current_index = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        flat_tensor[current_index:current_index+numel].mul_(tensor_weight).add_(parameter.data.mul(1-tensor_weight).view(-1))
        current_index += numel 

def copy_model_params(flat_tensor, model):
    '''copy model parameters into the flat tensor'''
    current_index = 0 
    for parameter in model.parameters():
        numel = parameter.data.numel()
        flat_tensor[current_index:current_index+numel].data.copy_(parameter.data.view(-1))
        current_index += numel

def copy_model_buffers(flat_tensor, model):
    '''copy model parameters into the flat tensor'''
    current_index = 0
    for parameter in model.buffers():
        numel = parameter.data.numel()
        flat_tensor[current_index:current_index+numel].data.copy_(parameter.data.view(-1))
        current_index += numel

def add_model_grads(flat_tensor, model):
    '''add grads of model parameters into the flat tensor'''
    current_index = 0 
    for parameter in model.parameters():
        numel = parameter.grad.data.numel()
        size = parameter.grad.data.size()
        flat_tensor[current_index:current_index+numel].add_(parameter.grad.data.view(-1))
        current_index += numel

def update_model_grads(p, flat_tensor, model):
    current_index = 0 
    for parameter in model.parameters():
        numel = parameter.grad.data.numel()
        size = parameter.grad.data.size()
        diff = parameter.data - flat_tensor[current_index:current_index+numel].view(size)
        parameter.grad.data.add_(p, diff)
        current_index += numel

def unravel_model_params(model, flat_tensor, is_grad, operation, model_weight=1):
    '''assign the flat tensor to model's parameters'''
    assert model_weight <=1 and model_weight >=0, 'weight coefficient should be in the range of [0,1]'
    current_index = 0
    if is_grad:   
        for parameter in model.parameters():
            numel = parameter.grad.data.numel()
            size = parameter.grad.data.size()
            if operation == 'add':
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.grad.data.mul_(model_weight)
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.grad.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 
    else:
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if operation == 'add':
                parameter.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.data.mul_(model_weight)
                parameter.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 

def unravel_model_buffers(model, flat_tensor, is_grad, operation, model_weight=1):
    '''assign the flat tensor to model's buffers'''
    assert model_weight <=1 and model_weight >=0, 'weight coefficient should be in the range of [0,1]'
    current_index = 0
    if is_grad:  
        for parameter in model.buffers():
            numel = parameter.grad.data.numel()
            size = parameter.grad.data.size()
            if operation == 'add':
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.grad.data.mul_(model_weight)
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.grad.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 
    else:
        for parameter in model.buffers():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if operation == 'add':
                parameter.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.data.mul_(model_weight)
                parameter.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 