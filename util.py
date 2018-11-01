import pickle as pk
import sys
import torchvision
from torchvision import transforms


############################################################
### IO
############################################################

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print ("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print ('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()
    
def unnormalize(y, mean, std):
    x = y.new(*y.size())
    x[:, 0, :, :] = y[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = y[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = y[:, 2, :, :] * std[2] + mean[2]
    return x

def data_mean_std(train_data_gen):
    pop_mean = []
    pop_std = []
    for inputs in train_data_gen:
        # shape (batch_size, 3, height, width)
        data , _ = inputs
        numpy_image = data.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std = np.std(numpy_image, axis=(0,2,3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    return pop_mean, pop_std