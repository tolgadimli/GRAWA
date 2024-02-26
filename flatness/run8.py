from typing import OrderedDict
import torch
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import os

import torch.nn as nn
import argparse
import logging

from lanczos import eig_trace
from utils import *
from model_class import model_for_sharp

parser = argparse.ArgumentParser()
_ = parser.add_argument('-f', '--foo', action='store_true')

# MAIN PROGRAM =================

args = parser.parse_args()
logging.basicConfig(filename = 'analysis_logger.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


# MAIN PROGRAM =================
device = 'cuda:0'
dataset_name = 'CIFAR100'
#subtrainset = CifarSubTrainSet()
#subdataloader = DataLoader(subtrainset, batch_size=128)


main_dir = '8 Worker'
model_dirs = ['densenet'] #, 'pyramidnet', 'wideresnet']

#all_train_loader, all_test_loader = get_cifar10_dataset()


valley_csv = []
for mdir in model_dirs:
    current_set = os.path.join(main_dir, mdir)
    all_opts_dirs = os.listdir(current_set) 
    train_loader, test_loader = get_data_loader(dataset_name, mdir, bs=32)

    if len(all_opts_dirs) != 0:
        logging.warning('=========================================================')
        logging.warning('Analyzing %s Experiments'%mdir)
        logging.warning('=========================================================')
        for opt_dir in all_opts_dirs:

            current_dir = os.path.join(current_set, opt_dir)
            seed_dirs = os.listdir(current_dir)
            if len(seed_dirs) != 0:

                logging.warning('Distributed Optimizer: %s'%opt_dir)
                seeds_eigen_max, seeds_eigen_trace = [], []
                
                for dir in tqdm(seed_dirs):
                       
                    load_dir =  os.path.join(current_dir, dir)
                    if len(os.listdir(load_dir)) != 0:

                        
                        if 'DP' in load_dir:
                            seed_number = int(load_dir.split('==')[-1])
                            logging.info("Current seed number: %d"%seed_number)
                            cw0 = os.path.join( load_dir, 'modelparams.pt' )
                        
                            w0 = torch.load(cw0)
                            w_avg = OrderedDict()
                            for key in w0:
                                new_key = key.split('module.')[-1]
                                w_avg[new_key] = w0[key]
                            
                
                        else:
                            seed_number = int(load_dir.split('-')[-1])
                            logging.info("Current seed number: %d"%seed_number)

                            if 'checkpoint-w0-end.pt' in os.listdir(load_dir):
                                cw0 = os.path.join( load_dir, 'checkpoint-w0-end.pt' )
                                cw1 = os.path.join( load_dir, 'checkpoint-w1-end.pt' )
                                cw2 = os.path.join( load_dir, 'checkpoint-w2-end.pt' )
                                cw3 = os.path.join( load_dir, 'checkpoint-w3-end.pt' )
                            else:
                                cw0 = os.path.join( load_dir, 'checkpoint-w4-end.pt' )
                                cw1 = os.path.join( load_dir, 'checkpoint-w5-end.pt' )
                                cw2 = os.path.join( load_dir, 'checkpoint-w6-end.pt' )
                                cw3 = os.path.join( load_dir, 'checkpoint-w7-end.pt' )            

                            w0 = torch.load(cw0)['state_dict']
                            w1 = torch.load(cw1)['state_dict']
                            w2 = torch.load(cw2)['state_dict']
                            w3 = torch.load(cw3)['state_dict']                   
         
                            w_avg = OrderedDict()
                            for key in w0:
                                w_avg[key] = (w0[key] + w1[key].to(device) + w2[key].to(device) + w3[key].to(device))/4.0

                        wor = get_model(mdir,dataset_name)
                        wor.load_state_dict( w_avg )
                        wor.to(device)
                        
                        # test_loss, test_error = get_model_eval(wor, test_loader, device)
                        criterion = nn.CrossEntropyLoss()
                        dset_loaders_dict = {}
                        dset_loaders_dict['train'] = train_loader #subdataloader
                        model_func = model_for_sharp(wor, dset_loaders_dict, criterion, True )

                        logging.info("Calculating eigen measures...")
                        max_eig, eigen_trace, eig15  = eig_trace(model_func, 100, draws=3, use_cuda=True, verbose=True) #100
                        logging.info("%f %f %f"%(max_eig, eigen_trace, eig15))
                        #print(max_eig, eigen_trace)

                    
                        v_csv = [ mdir, opt_dir, seed_number, max_eig, eigen_trace, eig15]
                        valley_csv.append(v_csv)

                        #logging.info('====================================================================')
                        print("Saving the experiment values to a csv.")
                        pd_valley = pd.DataFrame(valley_csv, columns = ["model", "dist_opt", "seed", "max_eig", "eigen_trace", "eig15" ] )
                        pd_valley.to_csv("results8.csv")
