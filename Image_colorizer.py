import os.path
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb

# Saved weights for the CNN model - Can be any of the trained weights matching Main generation 1.1.
# If other generations are required the Generator Class will need to be updated as well
PATH = './Trained_Models/Main_Gen1.1_Model_10Epochs_places365_valset_JB.pth'

# Color saturation multiplier
ColorSat_Multiplier = 2

# To use the generator, simply put .JPG images either in color or grayscale
# (Doesn't really matter as the grayscale layer is isolated anyway)
# into a folder in the same directory called "tobeGen" and supply a folder called "generated" in the same location.
# Then this program will loop through all images in "tobeGen" folder and put them in "generated" folder

if __name__ == '__main__':

    # If a cuda device is available, the training will prioritize this device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")


    # Generator is the class representing the architecture of the CNN. Is a child of PyTorch's nn.Module.
    # This generator is generation 1.1 and can as such only support Gen1.1 weights

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


    # Instantiation of the generator and loading the trained wieghts
    net = Generator().to(device)
    net.load_state_dict(torch.load(PATH))


    # Helper function for loading images and resizing them to be divisible by 8 - #todo fix clunky code
    def load_image(RGB_image_filename):
        RGB_array = imread(os.path.join('./tobeGen', RGB_image_filename))
        LAB_array = rgb2lab(RGB_array).astype('float32')
        width = np.size(RGB_array[:, 0, 0])
        height = np.size(RGB_array[0, :, 0])

        idx = 0
        idx2 = 0
        while (width / 2 / 2 / 2) % 1 != 0:
            width += 1
            idx += 1

            print(f'width: {width}, idx = {idx}')
            if idx >= 40:
                break
        while (height / 2 / 2 / 2) % 1 != 0:
            height += 1
            idx2 += 1

            print(f'height: {height}, idx = {idx2}')
            if idx2 >= 40:
                break

        img_size = width, height

        return RGB_array, LAB_array, img_size

    # Helper function to seperate L*-channel and A*B*-Channel.
    # Then forwarding through the model and contecating the layers to create a displayable RGB image
    # For more information either read the belonging raport of the commens of the main.py of image processing
    # TODO acutally write the rapport...
    def process_image(LAB_array, img_size):
        width, height = img_size
        L_array = LAB_array[..., 0]
        L_tensor = transforms.ToTensor()(L_array)
        L_tensor = transforms.Resize((width, height))(L_tensor)
        L_tensor = torch.unsqueeze(L_tensor, 0)
        L_array = L_tensor.numpy()
        Generated_AB_tensor = net.forward(L_tensor.to(device=device))

        Generated_AB_array = Generated_AB_tensor.cpu().detach().numpy() * ColorSat_Multiplier
        Merged_LAB_array = np.concatenate((L_array, Generated_AB_array), axis=1)
        Merged_LAB_array = Merged_LAB_array[0]
        Transposed_LAB_array = np.transpose(Merged_LAB_array, (1, 2, 0))
        Merged_RGB_array = lab2rgb(Transposed_LAB_array)

        return Merged_RGB_array

    # Simple dipslays helper function to see comparisons of original image and the generated image
    def display_image_pair(RGB_array, new_RGB_array):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(RGB_array)
        fig.add_subplot(1, 2, 2)
        plt.imshow(new_RGB_array)
        plt.show()

    # Img_save takes all of the images stored in folder "tobeGen"
    # and puts them through the model and stores them in folder "generated"

    def img_save(new_RGB_array, inp_filename):
        main_path = './generated'
        save_path = os.path.join(main_path, inp_filename)
        print(save_path)
        try:
            plt.imsave(save_path, new_RGB_array)
        except ValueError:
            print(f'{inp_filename} already has a generated picture in the folder')

    # Loop that goes through all the files in folder "tobeGen" and applies all the helper functions
    for filename in os.listdir('./tobeGen'):
        RGB_array, LAB_array, img_size = load_image(filename)

        new_RGB_array = process_image(LAB_array, img_size)

        display_image_pair(RGB_array, new_RGB_array)

        img_save(new_RGB_array, filename)
