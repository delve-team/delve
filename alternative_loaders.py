import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Transformations
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

# Downloading/Louding CIFAR10 data
trainset = CIFAR10(root='./data', train=True, download=True)  # , transform = transform_with_aug)
testset = CIFAR10(root='./data', train=False, download=True)  # , transform = transform_no_aug)
classDict = trainset.class_to_idx

# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets


# Define a function to separate CIFAR classes by class index

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


# ================== Usage ================== #

def get_n_fold_datasets_train(t, batch_size, class_names=['cat', 'dog']):

    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_trainset = \
        DatasetMaker(
            [get_class_i(x_train, y_train, classDict[class_names[0]]), get_class_i(x_train, y_train, classDict[class_names[1]])],
            transform_with_aug
        )

    kwargs = {'num_workers': 3, 'pin_memory': False}

    # Create datasetLoaders from trainset and testse

    trainsetLoader = DataLoader(cat_dog_trainset, batch_size=batch_size, shuffle=True, **kwargs)
    return trainsetLoader


def get_n_fold_datasets_test(t, batch_size, class_names=['cat', 'dog']):
    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_testset = \
        DatasetMaker(
            [get_class_i(x_test, y_test, classDict[class_names[0]]),
             get_class_i(x_test, y_test, classDict[class_names[1]])],
            transform_no_aug
        )

    kwargs = {'num_workers': 3, 'pin_memory': False}

    # Create datasetLoaders from trainset and testse

    testsetLoader = DataLoader(cat_dog_testset, batch_size=batch_size, shuffle=False, **kwargs)
    return testsetLoader