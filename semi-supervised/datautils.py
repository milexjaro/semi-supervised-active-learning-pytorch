from random import shuffle
import torch
import numpy as np
import sys
from urllib import request
from torch.utils.data import Dataset
from PIL import Image
sys.path.append("../semi-supervised")
n_labels = 10
cuda = torch.cuda.is_available()


class SpriteDataset(Dataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    * color
    * shape
    * scale
    * orientation
    * x-position
    * y-position
    """
    def __init__(self, transform=None):
        self.transform = transform
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        
        try:
            self.dset = np.load("./dsprites.npz", encoding="bytes")["imgs"]
        except FileNotFoundError:
            request.urlretrieve(url, "./dsprites.npz")
            self.dset = np.load("./dsprites.npz", encoding="bytes")["imgs"]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        sample = self.dset[idx]
                
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class SemiSupervisedActiveLearningDataset(Dataset):
    """Dataset for Semi-Supervised Active Learning experimentation"""
    def __init__(self, data_location, is_labelled=False, transform=None, target_transform=None, algorithm="", train=True, initial_number_of_data=None, data_size_cap=None):
        self.transform = transform
        self.target_transform = target_transform
        
        try:
            train_infix = 'train' if train else 'test'
            if train:
                labelled_infix = '_labelled' if is_labelled else '_unlabelled'
                algorithm_suffix = f'_{algorithm}' if algorithm else ''
                initial_number_of_data = f'_{initial_number_of_data}' if initial_number_of_data else ''
            else:
                labelled_infix = ''
                algorithm_suffix = ''
                initial_number_of_data = ''
            
            data_file_location = f"{data_location}/X_{train_infix}{labelled_infix}{algorithm_suffix}{initial_number_of_data}_base.pt"
            targets_file_location = f"{data_location}/y_{train_infix}{labelled_infix}{algorithm_suffix}{initial_number_of_data}_base.pt"

            print(data_file_location)
            print(targets_file_location)

            # Squeeze is needed because it has (x, 1, 28, 28 dimension)
            self.data = (torch.squeeze(torch.from_numpy(torch.load(data_file_location))) * 255).to(torch.uint8)
            self.targets = torch.from_numpy(torch.load(targets_file_location))

            if data_size_cap and train==True:
                if is_labelled:
                    self.data = self.data[:data_size_cap]
                    self.targets = self.targets[:data_size_cap]
                else:
                    unused_labelled_data_location = f"{data_location}/X_{train_infix}_labelled{algorithm_suffix}{initial_number_of_data}_base.pt"
                    self.unused_labelled_data = (torch.squeeze(torch.from_numpy(torch.load(unused_labelled_data_location))) * 255).to(torch.uint8)
                    self.data = torch.cat((self.data, self.unused_labelled_data[data_size_cap:]), 0)

                    unused_labelled_targets_location = f"{data_location}/y_{train_infix}_labelled{algorithm_suffix}{initial_number_of_data}_base.pt"
                    self.unused_labelled_targets = torch.from_numpy(torch.load(unused_labelled_targets_location))
                    self.targets = torch.cat((self.targets, self.unused_labelled_targets[data_size_cap:]), 0)
            print(len(self.data), len(self.targets))
        except FileNotFoundError:
            print("File is not found")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data


def get_mnist(location="./", batch_size=64, labels_per_class=10, algorithm=None, data_size_cap=None):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train_labelled = SemiSupervisedActiveLearningDataset(f'{location}', train=True, is_labelled=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels), algorithm=algorithm,
                        initial_number_of_data=n_labels*labels_per_class, data_size_cap=data_size_cap)
    mnist_train_unlabelled = SemiSupervisedActiveLearningDataset(f'{location}', train=True, is_labelled=False,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels), algorithm=algorithm,
                        initial_number_of_data=n_labels*labels_per_class, data_size_cap=data_size_cap)
    mnist_test = SemiSupervisedActiveLearningDataset(f'{location}', train=False,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train_labelled, batch_size=batch_size, num_workers=4, pin_memory=cuda,
                                           sampler=RandomSampler(mnist_train_labelled))
    unlabelled = torch.utils.data.DataLoader(mnist_train_unlabelled, batch_size=batch_size, num_workers=4, pin_memory=cuda,
                                           sampler=RandomSampler(mnist_train_unlabelled))
    test = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=RandomSampler(mnist_test))

    return labelled, unlabelled, test

def get_mnist_dataset(location="./", labels_per_class=100, algorithm=None):
    import torchvision.transforms as transforms
    from utils import onehot

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train_labelled = SemiSupervisedActiveLearningDataset(f'{location}', train=True, is_labelled=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels), algorithm=algorithm, initial_number_of_data=n_labels*labels_per_class)
    mnist_train_unlabelled = SemiSupervisedActiveLearningDataset(f'{location}', train=True, is_labelled=False,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels), algorithm=algorithm, initial_number_of_data=n_labels*labels_per_class)
    mnist_test = SemiSupervisedActiveLearningDataset(f'{location}', train=False,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    return mnist_train_labelled, mnist_train_unlabelled, mnist_test

def get_mnist_legacy_dataset(location="./"):
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    return mnist_train, mnist_valid



def get_mnist_legacy(location="./", batch_size=64, labels_per_class=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation