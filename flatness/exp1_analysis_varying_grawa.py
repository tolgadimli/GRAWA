import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
from scipy import stats
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

model = 'resnet18narrow_EASGD'
suffix = '_NOAUG_GRAWA'

exp_dir = '%s_figures%s'%(model, suffix)
dirs = os.listdir()
if exp_dir not in dirs:
    os.mkdir(exp_dir)

valleys = pd.read_csv("%s_valley%s.csv"%(model, suffix))
#valleys = pd.read_csv("%s_valley.csv"%(model))

#valleys = valleys.dropna(axis = 0, how = 'any')
valleys = valleys.fillna(1)
train_loss = valleys['avg_train_loss']
test_loss = valleys['avg_test_loss']
train_err = valleys['avg_train_error']
test_err = valleys['avg_test_error']

bs = valleys['batch_size']
mom = valleys['momentum']
lr = valleys['lr']
width = valleys['width']
wd = valleys['weight_decay']


print(len(test_err))

rem_inds = train_err[train_err < 1].index 

rem_inds = width[wd < 0.01 ].index
print(len(rem_inds))

train_loss = train_loss.loc[rem_inds]
test_loss = test_loss.loc[rem_inds]
train_err = train_err.loc[rem_inds]
test_err = test_err.loc[rem_inds]

gen_gap = test_err - train_err

metric_dict = {}

grawa = valleys['grawa'].loc[rem_inds]
mgrawa = valleys['mgrawa'].loc[rem_inds]

metric_dict["grawa"] = grawa
metric_dict["mgrawa"] = mgrawa



col_names = ["Measure","Corr."]
kendall_table = []
for key, value in metric_dict.items():
    row = []
    row.append(key)
    tau, p = stats.kendalltau( list(value), 20*np.log10(list(gen_gap)))
    tau = round(tau*1000)/1000
    row.append(tau)
    kendall_table.append(row)

pd_kendall = pd.DataFrame(kendall_table, columns=col_names)
csv_name = os.path.join(exp_dir, "%s_kendall_table.csv"%model)
pd_kendall.to_csv(csv_name)



# false_train_loss = center_train_loss.loc[false_inds]
# false_test_loss = center_test_loss.loc[false_inds]
# false_train_err = center_train_err.loc[false_inds]
# false_test_err = center_test_err.loc[false_inds]
# false_vss = valley_size.loc[false_inds]

# for l in range 
# xaxis_labels = []
measures = [grawa, mgrawa]
                #shannon_ent, eps_flat, frob_norm, fisher, -local_ent, lpf, entropy_grad, max_eig, eig_trace, pac]

labels = [  'Gradient Norm', 'mgrawa']
          
save_name = [ 'grawa', 'mgrawa']
           
for i, meas in enumerate(measures):


    meas_stand = (meas - np.min(meas)) / np.max(meas)
    print(type(meas_stand))

    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.title('EASGD-ResNet18 - Gen. Gap vs. %s'%labels[i])
    plt.xlabel(labels[i])
    plt.ylabel('Generalization Gap')
    plt.scatter(meas_stand, gen_gap, marker='*', c = 'r', linewidths=4)
    plt.grid()

    plt.subplot(1,2,2)
    plt.title('EASGD-ResNet18 - Gen. Gap vs. %s'%labels[i]) 
    plt.xlabel(labels[i] + ' (Standardized, log scale)')
    plt.ylabel('Generalization Gap')
    if i == 7 or i == 8:
        plt.xlim(-120, 5)
    
    plt.scatter(20*np.log10(meas_stand), gen_gap, marker='*', c = 'b', linewidths=4)
    plt.grid()
    save_dir = os.path.join(exp_dir, save_name[i])
    plt.savefig(save_dir)


# In text figure         
meas_stand = (grawa - np.min(grawa)) / np.max(grawa)
print(type(meas_stand))
plt.figure(figsize=(8,6))
plt.title('EASGD-ResNet18 - Gen. Gap vs. Gradient Norm')
plt.xlabel('Gradient Norm (standardized, log scale)')
plt.ylabel('Generalization Gap(%)')

plt.scatter(20*np.log10(meas_stand), gen_gap, marker='*', c = 'firebrick', linewidths=4)
plt.grid()
# save_dir = os.path.join(exp_dir, 'grad_norm_fig.png')
# plt.savefig(save_dir)