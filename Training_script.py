import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import os, time
from torch.utils.data import Dataset
import wandb


# These are the main variables to change if you don't want to poke around in the code.

# Batch_size is the amount of pictures per batch thats parsed through the model
batch_size = 4
# Epochs are the amount of times to go through the entire dataset
epochs = 1
# Learning_rate is the rate of which the optimizer is allowed to change the weights and biases
learning_rate = 0.001



# PATH should be changed for each training as to not overwrite previous tranings.
# Todo make automatic naming shceme
PATH = './Trained_Models/wandb_test1.pth'

# PATH is also the PATH that has to be used in generator to test the model on more images


# Composed transform layer for image augmentation
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(), ])

# Defining system paths to  locate images for custom dataset
image_path = './data/val_256_new'
train_image_paths = []
for pictures in os.listdir(image_path):
    train_image_paths.append(os.path.join(image_path, pictures))

# Supporting variables to store classes matching images
classes = []  # to store class values
idx_to_class = {i: j for i, j in enumerate(classes)}
class_to_idx = {value: key for key, value in idx_to_class.items()}


# Defining the custom dataset based on torchvision.Dataset Class so images can be dataloaded
class CustomDataset123(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    # getitem needs overhaul to allow for pickling, so it enables multiproccesing for dataloader
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert('RGB')
        label = image_filepath.split('/')[-2]
        label = list(class_to_idx[label])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Both of the training sets. CustomDataset123 contains 4.700 images, but requires aditional setup to run.
# Places365 is the generic dataset provided directly from torchvision and can be downloaded in-code.

trainset = CustomDataset123(image_paths=train_image_paths, transform=transform)
# trainset = torchvision.datasets.Places365(root='./data', split='val', small=True, download=False,
#                                           transform=transform)

# The loader works for both datasets, however num_workers has to be set to 0, for CustomDataset123,
# until pickling has been fixed
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

if __name__ == '__main__':
    # wandb.init(project="my-test-project", entity="mangus")
    # wandb.config = {
    #     "learning_rate": learning_rate,
    #     "epochs": epochs,
    #     "batch_size": batch_size
    # }

    # If a cuda device is avaliable, the traning will prioritize this device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Starting a timer to monitor training time
    start_time = time.time()

    # Before training starts, a batch of pictures matching the batch size is stored to view the subjective succes
    # of the image processing
    dataiter = (iter(trainloader))
    images, labels = dataiter.next()

    # Lots of IO to fix the transformations from arrays to tensors and Vice Versa
    original_set_tensor = images
    original_set_numpy = original_set_tensor.numpy()
    original_set_numpy = np.transpose(original_set_numpy, (2, 3, 0, 1))
    original_set_lab = rgb2lab(original_set_numpy)
    copy_of_lab = np.copy(original_set_lab).astype('float32')
    copy_of_lab = np.transpose(copy_of_lab, (2, 3, 0, 1))
    grayscale_copy_of_lab = copy_of_lab[:, 0]
    grayscale_copy_of_lab = torch.from_numpy(grayscale_copy_of_lab.astype('float32'))
    grayscale_copy_of_lab = torch.unsqueeze(grayscale_copy_of_lab, 1)


    # Generator is the class representing the architecture of the CNN. Is a child of PyTorch's nn.Module
    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(

                # Downsample
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                # Upsample
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2)
            )

        def forward(self, input):
            return self.model(input)

    # Instantiation of the generator
    gen = Generator().to(device)

    # Setting up both the criterion and optimizer such that they can be modfied. For this MSELoss and
    # PyTorch's Adam optimizer has been utilized
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)

    # Some variables to store loss
    total_loss = 0
    epoch_loss_list = []

    # Main Training loop. Epochs are based on value given to the variable
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):

            # Data is loaded from the dataloader, and labels are store in a variable with is just a placeholder.
            inp_img_inside, labels = data

            # Images goes through transformation from Tensor to numpy array
            original_set_tensor_inside = inp_img_inside
            original_set_numpy_inside = original_set_tensor_inside.numpy()
            original_set_numpy_inside = np.transpose(original_set_numpy_inside, (2, 3, 0, 1))
            # Then converting images from RGB to CIELAB
            original_set_lab_inside = rgb2lab(original_set_numpy_inside)
            # Then transposed back to tensor dimensions
            new_set_lab_inside = np.transpose(original_set_lab_inside, (2, 3, 0, 1))
            # A copy is then made of the full CIELAB spectrum, such that the L* channel can be used for input and
            # the A*B* channels can be used for MSELoss
            lab_grayscale_inside = np.copy(new_set_lab_inside)
            # L* channel
            lab_grayscale_inside = lab_grayscale_inside[:, 0]
            lab_AB_inside = np.copy(new_set_lab_inside)
            # A*B* Channels
            lab_AB_inside = lab_AB_inside[:, [1, 2]]

            lab_grayscale_inside = torch.from_numpy(lab_grayscale_inside.astype('float32'))
            # Unsqueezing L* channel to mimic 1 dimensional image channel
            lab_grayscale_inside = torch.unsqueeze(lab_grayscale_inside, 1)
            lab_target_inside = torch.from_numpy(lab_AB_inside.astype('float32'))

            # Resetting the optimizers weights
            optimizer.zero_grad()

            # Forwarding grayscale image trough the CNN
            output = gen(lab_grayscale_inside.to(device))

            # Calculating loss over output and the ground truth A*B* channels
            loss = criterion(output, lab_target_inside.to(device))

            # Backpropagation according to loss
            loss.backward()

            # Stepping the optimizer for improving model
            optimizer.step()

            # Variables that accumulate loss
            total_loss += loss.item()
            epoch_loss += loss.item()

            # Print loop to track progress of the training
            if i % 250 == 0:
                print(f'Epoch = {epoch}, I = {i},  Loss: {total_loss / 250}, Time: {time.time() - start_time}')
                total_loss = 0

            # #wandb integration
            # wandb.log({"loss": loss})
            #
            # # Optional
            # wandb.watch(gen)

        # Below is the code for plotting the training progress over epochs.
        epoch_loss_list.append(epoch_loss / 4700)
        plt.plot(epoch_loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Loss per Iteration')
        # If your python doesn't allow the script to continue before closing the figure windows.
        # (if you don't have SciView etc.)
        # Comment "plt.show()" out, so that it doesn't pause the training after each epoch
        plt.show()

    # After the training is done, the weights and bias' at the defined path
    torch.save(gen.state_dict(), PATH)

    # Image transformations to allow it to be displayed to monitor models performance
    GeneratedAB_img = gen(grayscale_copy_of_lab.to(device))
    GeneratedAB_img = GeneratedAB_img.cpu()
    GeneratedAB_img = GeneratedAB_img.detach().numpy()
    # concatenating the L* input channel with the A*B* output channels to create a 3-channel RGB
    merged_img = np.concatenate((grayscale_copy_of_lab, GeneratedAB_img), axis=1)

    fig = plt.figure(figsize=(10, 7))
    # Displays the groundtruth images
    for index in range(batch_size):
        fig.add_subplot(3, batch_size, index + 1)
        plt.axis('off')
        plt.imshow(original_set_numpy[:, :, index])
    # Displays the grayscale of the ground truth images
    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size + index + 1)
        plt.axis('off')
        plt.imshow(original_set_numpy[:, :, index, 0], cmap='gray')
    # Displays the generated colorization
    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size * 2 + index + 1)
        plt.axis('off')
        img_slice = merged_img[index]
        print_img = np.transpose(img_slice, (1, 2, 0))
        plt.imshow(lab2rgb(print_img))
    plt.show()
