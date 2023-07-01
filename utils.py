import torch 
import numpy as np
from os.path import exists
import os 
import torch.utils.data as data
from typing import List, Dict, Tuple
from torch import nn 
import matplotlib.pyplot as plt

# file_names: 
#   12000_train_mnistmnistmsvhnsynusps.npz
#   12000_test_mnistmnistmsvhnsynusps.npz
def get_data_loaders(filenames: Dict[str, str], batch_size: int = 128, init_seed = 11):
    """
    Args: 
        filenames: a dictionary containing filenames for train and test data,
                     {'train': train_file_name, 'test': test_file_name}
        batch_size: batch size for dataloaders
        init_seed: initial seed used for numpy and torch, used for reproducibility

    Returns: 
        full_dataloaders: a dictionary containing dataloaders for train, test and test_missing,
                            {'train': DataLoader for train, 'test': DataLoader for test, 'test_missing': DataLoader for test data with missing features (features filled with zeros)}
    """

    # set initial seeds for repreducibility 
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    print('datafiles to read: ', filenames)

    full_datasets = {'train': None, 'test': None, 'test_missing': None}
    full_dataloaders = {'train': None, 'test': None, 'test_missing': None, 'train_size': None, 'test_size': None, 'test_missing_size': None}

    for phase in ['train', 'test', 'test_missing']:
        if phase == 'test_missing': 
            imgs, digit_labels, domain_labels, features, num_domains = load_ds(filenames['test'], 'test', features_missing = True)
        else:
            imgs, digit_labels, domain_labels, features, num_domains = load_ds(filenames[phase], phase)

        full_datasets[phase] = customDataset(imgs, domain_labels, digit_labels, features)
        full_dataloaders[phase] = data.DataLoader(full_datasets[phase], batch_size=batch_size, shuffle=True) # (phase=='train')
        full_dataloaders[f'{phase}_size'] = full_datasets[phase].__len__()

    return full_dataloaders, num_domains



def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

class customDataset(data.Dataset):
    def __init__(self, data, domain_labels, digit_labels, features) -> None:
        super().__init__()
        self.data = data
        self.domain_labels = domain_labels
        self.digit_labels = digit_labels
        self.features = features
    
    def __getitem__(self, index):
        """
        return data, features, domain label, digit label
        """
        # ---> changed it
        return torch.FloatTensor(self.data[index]), \
                torch.FloatTensor(self.features[index]), \
                    torch.tensor(self.domain_labels[index]).type(torch.int64), torch.tensor(self.digit_labels[index]).type(torch.int64)
        
    def __len__(self): 
        return self.data.shape[0]
    
def load_ds(filename: str, phase: str, features_missing: bool = False):
    """
    phase either 'train' or 'test' or 'test_missing'
    """
    data = np.load(f'{filename}')
    imgs = data[f'{phase}_imgs']
    digit_labels = data[f'{phase}_digit_labels']
    domain_labels = data[f'{phase}_domain_labels']
    if features_missing: 
        features = np.zeros_like(data[f'{phase}_features'])
        # features = np.empty_like(data[f'{phase}_features'], dtype=np.int32)

    else:    
        features = data[f'{phase}_features']

    print(f'reading {filename}, number of samples: {imgs.shape[0]}')

    return imgs, digit_labels, domain_labels, features, np.unique(domain_labels).shape[0]



def custpm_plot_loss(loss_hist, phase_list, loss_name: str, title: str, dir: str): 
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=[7, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/loss_{loss_name}.jpg')
    plt.clf()

def custom_plot_training_stats(acc_hist, loss_hist, phase_list, title: str, dir: str): 
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=[14, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    # acc: 
    for phase in phase_list:
        highest_acc_x = np.argmax(np.array(acc_hist[phase]))
        highest_acc_y = acc_hist[phase][highest_acc_x]
        
        ax2.annotate("{:.4f}".format(highest_acc_y), [highest_acc_x, highest_acc_y])
        ax2.plot(acc_hist[phase], '-x', label=f'{phase} loss', markevery = [highest_acc_x])

        ax2.set_xlabel(xlabel='epochs')
        ax2.set_ylabel(ylabel='acc')

        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax2.legend()
        #ax2.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/acc_loss.jpg')
    plt.clf()