import time
import copy
import torch

from random import randint

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from local_tools import *
from dist_tools import *
from worker_base import WorkerBase

class WorkerDistOptim(WorkerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        '''class for distribured optimization'''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

        # create groups
        if cur_worker % args.num_gpus == 0:
            print("\n=== Setting up distribution training environment ===")
        dist.init_process_group(rank=self.my_rank, world_size=args.world_size, 
                                backend=args.dist_backend, init_method=args.dist_url_wrk)

        self.group_dict = {}
        for i in range(args.num_groups):
            _group = [j for j in range(self.args.cur_group* self.args.num_gpus, (self.args.cur_group+1)* self.args.num_gpus)]
            self.group_dict[i] = torch.distributed.new_group(_group)
        
        # barrier test
        dist_print(args, "--[Barrier] Now entering barrier test!!")
        dist_dumb_barrier(args, self.device)

        # distributed data_loader 
        dist_print(args, "--[Loader] Preparing data sets and data loaders")
        self.train_sampler, self.train_dataset, self.train_loader,\
            self.test_dataset, self.test_loader = get_data_loader(args)

        self.all_trainset = get_all_trainset(args)
        ''' debugging: print('Sampler Test', self.train_sampler == self.train_loader.sampler) '''

        # obtain loaders' sizes
        self.train_dataset_size = len(self.train_dataset)
        self.train_loader_size = len(self.train_loader)
        self.test_dataset_size = len(self.test_dataset)
        self.test_loader_size = len(self.test_loader)

        # distributed tensors
        self.grawa_pull_list = [0.1*i for i in range(1,10)]
        dist_print(args, "--[Variables] Initilizing distribution variables")
        self.dist_params_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)
        self.dist_grads_tensor = ravel_model_params(self.model, is_grad=True, device=self.device)
        

        #args.size = 4 * args.batch_size # in the unit of number of samples
        # if args.dataset == 'CIFAR10':
        #     args.size = 512 # in the unit of number of samples
        # elif args.dataset == 'ImageNet':
        args.size = 1 * args.batch_size
        print(args.size)
        s = time.time()
        
        self.subdataloader = get_subdataloader( self.args, self.train_dataset )
        print(time.time() - s)

        self.num_layers =  get_model_layer_num(self.model)
        if args.dist_optimizer == 'LGRAWA':
            self.dist_layerwise_grad_norms = torch.zeros( self.num_layers  ).to(self.device)
        elif args.dist_optimizer == 'MGRAWA':
            self.dist_layerwise_grad_norms = torch.zeros( 1 ).to(self.device)

        # GRAWA momentum
        self.grawa_momentum = self.args.grawa_momentum
        self.gn_momentum = 0.9**8 # this value is inherited from EASGD and roughly equals to 0.43

        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=0, async_op=False)
        self.dist_group_params_tensor = self.dist_params_tensor.clone()
        
        # self.dist_buffers_tensor = ravel_model_buffers(self.model, is_grad=False, device=self.device)
        # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)

        # same model initialization
        unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='copy', model_weight=0)
        self.dist_params_tensor_msr = self.dist_params_tensor.clone()
        self.dist_params_tensor_prop = self.dist_params_tensor.clone()

        # model buffers
        # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy', model_weight=0)

        if self.my_rank % self.args.num_gpus == 0:
            print('The parameter tensor has length of %d and size of %.3f MB'%(len(self.dist_params_tensor), 32* (1.25e-7)* len(self.dist_params_tensor)))
            # print('The buffer tensor has length of %d and size of %.3f MB'%(len(self.dist_buffers_tensor)), 32* (1.25e-7)* len(self.dist_buffers_tensor))
            print('\n=== Proceeding to main subprocess===')
        
        # counter
        self.l_comm_counter = 0
        self.g_comm_counter = 0
        self.dist_cur_lr = self.args.lr

        message_size = 3
        # define other necessary tensors
        if args.dist_optimizer == 'LSGD':
            message_size = 3
        elif args.dist_optimizer == 'LdLGRAWA':
            #print(get_model_layer_num(self.model))
            message_size = 2 + self.num_layers 
            #print(message_size)
        self.dist_message_tensor = torch.zeros(message_size, device=self.device) #| rank | loss | LR |
        self.dist_message_list = [torch.zeros(message_size, device=self.device) for i in range(args.world_size)]

    def dist_lsgd_train(self):
        if self.args.landscape:
            _model_params_tensor = self.dist_group_params_tensor.clone()
            copy_model_params(_model_params_tensor, self.model)

        # 1. update messages
        self.l_comm_counter += 1  
        self.dist_message_tensor[0] = self.my_rank
        self.dist_message_tensor[1] = self.cur_lr
        self.dist_message_tensor[2] = self.local_train_loss
        self.dist_params_req = dist.all_gather_multigpu([self.dist_message_list], [self.dist_message_tensor], async_op=False)
        
        
        # 2.1 update parameters and buffers locally
        if self.args.is_lcomm:
            # local-parameters
            cur_group_message = self.dist_message_list[self.args.cur_group* self.args.num_gpus: (self.args.cur_group+1)* self.args.num_gpus]
            l_leader = min(cur_group_message, key = lambda x: x[2].item())
            l_leader_rank = int(l_leader[0].item())
            assert self.args.cur_group* self.args.num_gpus <= l_leader_rank and l_leader_rank < (self.args.cur_group+1)* self.args.num_gpus
            self.world_best_worker = l_leader_rank
            if self.my_rank == l_leader_rank:
                copy_model_params(self.dist_group_params_tensor, self.model)
            self.dist_params_req = dist.broadcast_multigpu([self.dist_group_params_tensor], src = l_leader_rank, group = self.group_dict[self.args.cur_group], async_op=False)
            unravel_model_params(self.model, self.dist_group_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c1)

            # local-buffers
            # copy_model_buffers(self.dist_buffers_tensor, self.model)
            # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
            # if self.my_rank == 0:
            #     self.dist_buffers_tensor.div_(self.args.num_gpus)
            # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
            # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')
        
        # 3. update parameters and buffers globally
        gl_prop = int(max(self.args.g_comm//self.args.l_comm, 1))
        if self.l_comm_counter % gl_prop == 0:
            self.g_comm_counter += 1
            # global-parameters
            g_leader = min(self.dist_message_list, key = lambda x: x[2].item())
            g_leader_rank = int(g_leader[0].item())
            assert 0 <= g_leader_rank and  g_leader_rank < self.args.num_groups* self.args.num_gpus
            self.world_best_worker = g_leader_rank
            if self.my_rank ==  g_leader_rank:
                copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=g_leader_rank, async_op=False)
            unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c2) 

            # global-buffers
            # copy_model_buffers(self.dist_buffers_tensor, self.model)
            # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = 0, async_op=False)
            # if self.my_rank == 0:
            #     self.dist_buffers_tensor.div_(self.args.world_size)
            # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)
            # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')
            
    def dist_lsgd_test(self):
        if self.args.is_lcomm:
            copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.all_reduce_multigpu([self.dist_params_tensor], group = self.group_dict[self.args.cur_group], async_op=False)
            self.dist_params_tensor.div_(self.args.num_gpus)
            unravel_model_params(self.model_center, self.dist_params_tensor, is_grad=False, operation='copy')
            # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
            test_loss, test_error = self.local_center_test()
        else:
            copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.all_reduce_multigpu([self.dist_params_tensor], async_op=False)
            self.dist_params_tensor.div_(self.args.world_size)
            unravel_model_params(self.model_center, self.dist_params_tensor, is_grad=False, operation='copy')
            # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
            test_loss, test_error = self.local_center_test()
        return test_loss, test_error
                            
    def dist_easgd_train(self):
        # [easgd] step 1: master gather current information from all workers step 2: master broadcast its parameters from last communication period 
        # The tricky part for EASGD is that: the update of master involves workers x_t instrad of x_{t+1}
        # self.args.beta = 0.43
        self.args.beta = self.args.c2
        copy_model_params(self.dist_params_tensor, self.model) # get x_i
        # if self.my_rank == 1:
        #     print(self.dist_params_tensor)
        self.dist_params_req = dist.reduce_multigpu([self.dist_params_tensor], dst = 0, async_op=False) # sum all x_i's
        #print(self.dist_params_req)
        # if self.my_rank == 1:
        #     print(self.dist_params_tensor)

        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor_msr], src = 0, async_op=False) # x tilde
        # if self.my_rank == 1:
        #     print(self.dist_params_tensor)
        
        # [easgd] master and workers will be pulled to each other: (a) workers pull to master (b) master pulls to workers
        # The update comes from easgd paper equations (5) and (6)
        alpha = self.args.beta/self.args.world_size        
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha) # x_i = x_i*(1-alpha) + x_tilde * alpha
        if self.my_rank == 0:
            self.dist_params_tensor.div_(self.args.world_size) # average of all x_i's
            self.dist_params_tensor_msr.mul_(1-self.args.beta) # x_tilde = x_tilde * (1-world_size*alpha)
            self.dist_params_tensor_msr.add_(self.args.beta, self.dist_params_tensor)   # x_tilde = x_tilde + [ alpha*(x_i) for all i = 1:world_size ]
                                                                                        # equivalent to    x_tilde = x_tilde + beta*(x)  
               
    def dist_easgd_test(self):
        # test with the averaged parameters
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor_msr], src = 0, async_op=False)
        unravel_model_params(self.model_center, self.dist_params_tensor_msr, is_grad=False, operation='copy')
        # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
        test_loss, test_error = self.local_center_test()
        return test_loss, test_error

    def dist_mgrawa_train(self):
        #self.subdataloader = get_subdataloader( self.args, self.train_dataset )
        self.g_comm_counter += 1
        subdataloader = get_subdataloader( self.args, self.all_trainset )
        current_grad_norms = get_grad_norm_tensor(self.local_optimizer, self.model, subdataloader, self.device, level = 'model').to(self.device)

        self.dist_layerwise_grad_norms = self.gn_momentum * self.dist_layerwise_grad_norms + (1 - self.gn_momentum) * current_grad_norms
        self.dist_layerwise_grad_norms_est = self.dist_layerwise_grad_norms / (1 - self.gn_momentum ** self.g_comm_counter)

        grawa_weights = self.dist_layerwise_grad_norms_est.pow(-1) 
        self.dist_layerwise_grad_norms_total = grawa_weights.clone() 
        _ = dist.all_reduce_multigpu([self.dist_layerwise_grad_norms_total], async_op=False)
        grawa_weights.div_(self.dist_layerwise_grad_norms_total) # normalized between 0 and 1
        
        copy_weighted_model_params( self.dist_params_tensor_msr, self.model, grawa_weights, 'model' )
        _ = dist.all_reduce_multigpu([self.dist_params_tensor_msr], async_op=False)
    
        alpha = self.args.c2
        #alpha = self.grawa_pull_list[randint( 0, len(self.grawa_pull_list)-1 )]
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha) 
        
        
    def dist_lgrawa_train(self):
        #start_time = time.time()
        self.g_comm_counter += 1
        subdataloader = get_subdataloader( self.args, self.all_trainset )
        current_grad_norms = get_grad_norm_tensor(self.local_optimizer, self.model, subdataloader, self.device, level = 'layer').to(self.device)
        #self.dist_layerwise_grad_norms.div( torch.ones().to(self.device) )

        self.dist_layerwise_grad_norms = self.gn_momentum * self.dist_layerwise_grad_norms + (1 - self.gn_momentum) * current_grad_norms
        self.dist_layerwise_grad_norms_est = self.dist_layerwise_grad_norms / (1 - self.gn_momentum ** self.g_comm_counter)
        
        grawa_weights = self.dist_layerwise_grad_norms_est.pow(-1)
        self.dist_layerwise_grad_norms_total = grawa_weights.clone()
        _ = dist.all_reduce_multigpu([self.dist_layerwise_grad_norms_total], async_op=False)
        grawa_weights.div_(self.dist_layerwise_grad_norms_total) # normalized between 0 and 1
        
        copy_weighted_model_params( self.dist_params_tensor_msr, self.model, grawa_weights, 'layer' )
        _ = dist.all_reduce_multigpu([self.dist_params_tensor_msr], async_op=False)
    
        alpha = self.args.c2
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha) 


    def dist_mgrawa_test(self):
        # test with the averaged parameters
        copy_model_params(self.dist_params_tensor, self.model)
        _ = dist.reduce_multigpu([self.dist_params_tensor], dst = 0, async_op=False) # op = dist.ReduceOp.AVG
        _ = dist.broadcast_multigpu([self.dist_params_tensor], src = 0, async_op=False)
        unravel_model_params(self.model_center, self.dist_params_tensor/self.args.num_gpus, is_grad=False, operation='copy')
        # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
        test_loss, test_error = self.local_center_test()
        return test_loss, test_error

    def dist_lgrawa_test(self):
        # test with the averaged parameters
        copy_model_params(self.dist_params_tensor, self.model)
        _ = dist.reduce_multigpu([self.dist_params_tensor], dst = 0, async_op=False) # op = dist.ReduceOp.AVG
        _ = dist.broadcast_multigpu([self.dist_params_tensor], src = 0, async_op=False)
        unravel_model_params(self.model_center, self.dist_params_tensor/self.args.num_gpus, is_grad=False, operation='copy')
        # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
        test_loss, test_error = self.local_center_test()
        return test_loss, test_error

    def dist_train(self):
        ''' distributed training '''
        # ============== WORK IN PROGRESS (***) ================
        # assign the values of exponetial averaging tensor mu to current model parameters
        # if self.args.weight_averaging:
        #     unravel_model_params(self.model, self.local_mu_tensor, is_grad=False, operation='copy')
        if self.args.weight_averaging:
            self.local_x_tensor.mul_(1-self.args.etagamma).add_(self.args.etagamma, self.local_mu_tensor)
            unravel_model_params(self.model, self.local_x_tensor, is_grad=False, operation='copy')
        # ============== WORK IN PROGRESS (***) ================

        if self.args.dist_optimizer == 'LSGD':
            self.dist_lsgd_train()

        elif self.args.dist_optimizer == 'EASGD':
            self.dist_easgd_train()

        elif self.args.dist_optimizer == 'LGRAWA':
            self.dist_lgrawa_train()

        elif self.args.dist_optimizer == 'MGRAWA':
            self.dist_mgrawa_train()   

        elif self.args.dist_optimizer == 'MGRAWAWrapper':
            self.dist_mgrawa_wrapper_train()
        
        elif self.args.dist_optimizer == 'LGRAWAWrapper':
            self.dist_lgrawa_wrapper_train()

        # ============== WORK IN PROGRESS (****) ===============
        # assign current model parameters' values to exponetial averaging tensor mu
        if self.args.weight_averaging:
            copy_model_params(self.local_x_tensor, self.model)
            self.local_mu_tensor.copy_(self.local_x_tensor)
        # ============== WORK IN PROGRESS (****) ===============

    def dist_test(self):
        ''' distributed testing '''
        if self.args.dist_optimizer == 'LSGD':
            test_loss, test_error = self.dist_lsgd_test()

        elif self.args.dist_optimizer == 'EASGD':
            test_loss, test_error = self.dist_easgd_test()

        elif self.args.dist_optimizer == 'LGRAWA':
            test_loss, test_error = self.dist_lgrawa_test()

        elif self.args.dist_optimizer == 'MGRAWA':
            test_loss, test_error = self.dist_mgrawa_test()

        elif 'GRAWAWrapper' in self.args.dist_optimizer:
            test_loss, test_error = self.dist_mgrawa_test()
        # not necessary but for safety we'd better zero out the distributed tensors
        self.dist_params_tensor.zero_()
        return test_loss, test_error


    # Experimental
    def dist_mgrawa_wrapper_train(self):
        #self.subdataloader = get_subdataloader( self.args, self.train_dataset )

        layerwise_grad_norms = self.local_optimizer.grawa_grad_norm_est
        # print(layerwise_grad_norms)
        #print(self.local_optimizer.grawa_grad_norm_list)
        mgrawa_score = torch.Tensor([sum(layerwise_grad_norms).item()]).to(self.device)
        # print(mgrawa_score)

        grawa_weights = mgrawa_score.pow(-1)
        # print(grawa_weights)
        self.dist_layerwise_grad_norms_total = grawa_weights.clone()
        _ = dist.all_reduce_multigpu([self.dist_layerwise_grad_norms_total], async_op=False)

        # print(grawa_weights)
        grawa_weights.div_(self.dist_layerwise_grad_norms_total) # normalized between 0 and 1
        # print(grawa_weights)

        # message_file = open(self.weights_file_name, "a")
        # _text = "\n%d,%.3f"%(self.my_rank, grawa_weights.item())
        # message_file.write(_text)
        # message_file.close()

        copy_weighted_model_params( self.dist_params_tensor_msr, self.model, grawa_weights, 'model' )
        _ = dist.all_reduce_multigpu([self.dist_params_tensor_msr], async_op=False)
    
        alpha = self.args.c2
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha) 

        # print(self.local_optimizer.state["step"])
        self.local_optimizer.reset()
        # print(self.local_optimizer.state["step"])
        # print('resetted!')



    # Experimental
    def dist_lgrawa_wrapper_train(self):
        #self.subdataloader = get_subdataloader( self.args, self.train_dataset )

        layerwise_grad_norms = self.local_optimizer.grawa_grad_norm_est
        lgrawa_score = layerwise_grad_norms.to(self.device)
        # print(lgrawa_score)

        grawa_weights = lgrawa_score.pow(-1)
        # print(grawa_weights)
        self.dist_layerwise_grad_norms_total = grawa_weights.clone()
        _ = dist.all_reduce_multigpu([self.dist_layerwise_grad_norms_total], async_op=False)

        # print(grawa_weights)
        grawa_weights.div_(self.dist_layerwise_grad_norms_total) # normalized between 0 and 1
        # print("Weights:", grawa_weights)


        msg_weights_content = ["%.2f"%gw.item() for gw in grawa_weights]
        msg_weights_content = "\n%d,"%self.my_rank + ",".join(msg_weights_content)
        # print(msg_weights_content)

        # message_file = open(self.weights_file_name, "a")
        # _text = msg_weights_content
        # message_file.write(_text)
        # message_file.close()
        
        copy_weighted_model_params( self.dist_params_tensor_msr, self.model, grawa_weights, 'layer' )
        _ = dist.all_reduce_multigpu([self.dist_params_tensor_msr], async_op=False)
    
        alpha = self.args.c2
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha) 