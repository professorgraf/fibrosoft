
import os
import random
import time
import gc
import math
import sys
import copy

import numpy as np
from PIL import Image

from adjustableimagecanvas import Marker
from numpy.ma.core import masked

from machinelearning import mlprocessing
from machinelearning import mlexceptions

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as Ftrans
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from torch.nn.functional import relu


# PARAMETERS and SETTINGS for the UNET behaviour control

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
BATCH_SIZE = 40
EARLY_STOP_PATIENCE = 5
INIT_LR = 0.00045
NUM_EPOCHS = 2000
LOSS_FCT = ['CE', None]   # ('DICE', 'CE_W')    # 'CE' / 'BCE' / 'DICE' / 'L1' / 'CE_W' CE with weights... or NoLoss
THRESHOLD = 200
LOSS_WEIGHTS = torch.Tensor( [0.3, 0.15, 0.55] )
ENC_CHANNELS = (3, 64, 128, 256, 512, 1024)
DEC_CHANNELS = (1024, 512, 256, 128, 64)


class MulticlassDiceLoss(nn.Module): # WARNING!
    # problematic!!! did not really work well
    ### NEEDS targets as one hot encoded tensors!
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """

        # print(f"Dice Score: shape1: {logits.shape}, shape2 {targets.shape}")
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(logits)
        # end if
        targets_one_hot = targets
        #targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        #print(targets_one_hot.shape)
        ## Convert from NHWC to NCHW
        #targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # predicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        if intersection < 0:
            intersection = -intersection

        #mod_a = logits.numel()  # intersection.sum()
        # mod_b = targets.numel()
        mod_a = math.fabs(logits.sum())
        mod_b = math.fabs(targets.sum())

        # DEBUG MSG: print(f"{intersection} *2 / ({mod_a}+{mod_b} --- what if {logits.numel()}")

        dice_coefficient = ( 2. * intersection / (mod_a + mod_b + smooth) + smooth)
        dice_loss = -dice_coefficient.log()
        if math.isinf(dice_loss) or math.isnan(dice_loss):
            print("ERROR: ")
            print(f"{intersection} *2 / ({mod_a}+{mod_b} --- what if {logits.numel()}")
            print(f"smoothing is {smooth}")

        return dice_loss


class NoLoss(nn.Module):
    def forward(self, logits, targets, reduction='mean'):
        return 0


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        # added batch norm
        # Sergey Ioffe, Christian Szegedy ‘Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift’
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() # nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()  # added for completion as well
        # nn.LeakyReLU()

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        # return self.conv2(self.relu(self.conv1(x)))

        # added batch-norm call as suggested - as well as a second relu
        return self.relu2(self.batch_norm2(self.conv2(self.relu(self.batch_norm(self.conv1(x))))))
        # return self.relu2(self.conv2(self.relu(self.conv1(x))))   # line without batchnorm


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        # maxpooling (before next layer = after each block of this previous list, see in forward)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        outputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            outputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return outputs


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, enc_features):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            x = self.enlarge(x, enc_features[i].shape)
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    @staticmethod
    def enlarge(features, shape):
        return F.interpolate(features, shape[2:])

    @staticmethod
    def crop(enc_features, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        # return the cropped features
        return enc_features


class UNet(nn.Module):
    # UNET class constructor
    def __init__(self, enc_channels=(3, 16, 32, 64), dec_channels=(64, 32, 16),
                 nb_classes=3, retain_dim=True,
                 out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)):

        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(dec_channels[-1], nb_classes, 1)
        self.batchnorm_out = nn.BatchNorm2d(nb_classes)
        self.retainDim = retain_dim
        self.outSize = out_size

    def forward(self, x):
        # grab the features from the encoder
        enc_features = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[::-1][0],
            enc_features[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        segmentation_map = self.batchnorm_out(self.head(dec_features))
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            segmentation_map = F.interpolate(segmentation_map, self.outSize)
        # return the segmentation map
        return segmentation_map


class UnsupervisedMaskedTrainingDataset(Dataset):
    def __init__(self, image_paths, transforms, masking_ratio=0.01, neighborhood=0):
        self.imagePaths = image_paths
        self.transforms = transforms
        self.masking_ratio = masking_ratio
        self.neighborhood = neighborhood

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image_path = self.imagePaths[idx]
        self.masking_ratio /= 100
        # load the image from disk
        image = Image.open(image_path)                                  # input image will be destroyed
        label = image.copy()                                            # label image will be kept in its original version
        noptbm = int(image.width * image.height * self.masking_ratio)   # number of pixel positions to be modified
        width = image.width
        height = image.height

        black = 0
        if image.mode == 'RGB':
            black = (0, 0, 255)
        elif image.mode == 'RGBA':
            black = (0, 0, 0, 255)
        for i in range(0, noptbm):
            x = random.randint(0, width-1)
            y = random.randint( 0, height-1)
            while image.getpixel((x,y)) == black:
                x = random.randint(0, width-1)
                y = random.randint(0, height-1)
            if self.neighborhood > 0:
                for dy in range(y-self.neighborhood, y+self.neighborhood):
                    for dx in range(x-self.neighborhood, x+self.neighborhood):
                        if 0 <= dx < width and 0 <= dy < height:
                            image.putpixel((dx, dy), black)
            elif 0 <= x < width and 0 <= y < height:
                image.putpixel((x, y), black)

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        # return modified image as image and original image as label --> we try to reconstruct
        return image, label


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.imagePaths = image_paths
        self.maskPaths = mask_paths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.imagePaths[idx]
        mask_path = self.maskPaths[idx]
        # load the image from disk
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        # return a tuple of the image and its mask
        return image, mask


class EarlyStopper_SingleVal:
    def __init__(self, patience=EARLY_STOP_PATIENCE, delta=0.000001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.patience_counter = 0

    def __call__(self, loss_value):
        if self.best_loss is None:
            self.best_loss = loss_value
        elif self.best_loss - self.delta < loss_value:
            self.patience_counter += 1
            if self.patience_counter > self.patience:
                return True
        else:
            self.patience_counter = 0
            self.best_loss = loss_value

        return False


class EarlyStopper:
    def __init__(self, patience=EARLY_STOP_PATIENCE, delta=1e-6, len_sequence=5):
        """
        Args:
        patience (int): Wie viele Versuche ohne Verbesserung zugelassen sind, bevor das Training gestoppt wird.
        delta (float): Der minimale Unterschied, um als Verbesserung zu zählen.
        len_sequence (int): Anzahl der aufeinander folgenden Loss-Werte, deren Durchschnitt verglichen wird.
        """
        self.patience = patience
        self.delta = delta
        self.len_sequence = len_sequence
        self.losses = []
        self.patience_counter = 0
        self.lowest = None
        self.end_next = False

    def __call__(self, loss_value):
        # Neuen Loss-Wert zur Liste hinzufügen
        self.losses.append(loss_value)

        # Wenn wir nicht genug Loss-Werte haben, um eine Sequenz zu bilden, kann noch nichts verglichen werden
        if len(self.losses) < 2 * self.len_sequence:
            return False

        # Die letzten len_sequence Werte und die davor liegenden len_sequence Werte nehmen
        prev_mean = sum(self.losses[-2 * self.len_sequence: -self.len_sequence]) / self.len_sequence
        curr_mean = sum(self.losses[-self.len_sequence:]) / self.len_sequence
        if self.lowest is None:
            self.lowest = prev_mean
        elif self.lowest > prev_mean:
            self.lowest = prev_mean

        # Prüfen, ob der aktuelle Mittelwert eine Verbesserung darstellt
        if prev_mean - curr_mean < self.delta:
            self.patience_counter += 1
            if not self.end_next:
                print(f"[DEBUG] slowly losing patience... {self.patience_counter} of {self.patience}")
        else:
            self.patience_counter = 0

        # print(f"[DEBUG] checking directory {os.curdir} for file early.stop")
        if os.path.exists('./early.stop'):
            os.remove('./early.stop')
            if self.end_next:
                return True
            self.end_next = True

        # check if patience ended, then stop next time we reach lowest value
        if self.patience_counter >= self.patience:
            self.end_next = True

        if self.end_next and loss_value <= self.lowest:
            return True

        return False


class UNetProcessing(mlprocessing.MLProcessing):
    def __init__(self, classes=3, epochs=NUM_EPOCHS, learning_rate=INIT_LR, optimizer='SGD', train_on_gpu=True,
                 visualize_linear=False, unsupervised=False):
        super().__init__()
        self.training_dataset = None
        self.test_dataset = None
        self.model = None
        self.train_on_gpu = train_on_gpu
        self.use_scheduler = False
        self.visualize_linear = visualize_linear
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.training_loader = None
        self.test_loader = None
        self.optimizer_name = optimizer
        self.unsupervised = unsupervised
        self.classes = classes
        self.Hist = {"train_loss": [0], "test_loss": [0]}

        self.device = "cpu"
        if self.train_on_gpu and torch.cuda.is_available():
            self.device = "cuda:0"
        elif self.train_on_gpu and torch.backends.mps.is_available():
            self.device = "mps"
        print(f"[INFO] working on {self.device}")

        # enc_channels=(3, 64, 128, 256), dec_channels=(256, 128, 64)
        self.model = UNet(enc_channels=ENC_CHANNELS, dec_channels=DEC_CHANNELS,
                          nb_classes=classes, retain_dim=True, out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        self.model.to(self.device)
        print(f"[DEBUG] having features like {DEC_CHANNELS}")
        self.log(f"[DEBUG] having features like {DEC_CHANNELS}")

        if self.classes == 1:
            print("1 class ---> default to DICE + no other")
            LOSS_FCT[0] = "DICE"
            LOSS_FCT[1] = "NoLoss"   # BCE

        self.loss_function = [None] * len(LOSS_FCT)
        for lfct_index in range(len(LOSS_FCT)):
            if LOSS_FCT[lfct_index] == 'BCE':
                self.loss_function[lfct_index] = torch.nn.BCEWithLogitsLoss()
            elif LOSS_FCT[lfct_index] == 'CE':
                self.loss_function[lfct_index] = torch.nn.CrossEntropyLoss()
            elif LOSS_FCT[lfct_index] == 'CE_W':
                self.loss_function[lfct_index] = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(self.device))
            elif LOSS_FCT[lfct_index] == "DICE":
                print("[WARN] currently Multiclass Dice is errorenous.")
                self.loss_function[lfct_index] = MulticlassDiceLoss(num_classes=self.classes) ##, softmax_dim=1)
            elif LOSS_FCT[lfct_index] == 'NLL':
                self.loss_function[lfct_index] = torch.nn.NLLLoss()
            elif LOSS_FCT[lfct_index] == 'L1':
                self.loss_function[lfct_index] = torch.nn.L1Loss()
            else:
                self.loss_function[lfct_index] = NoLoss()    # skip a loss

        self.optimizer = None
        self.scheduler = None

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_visualize_linear(self, linear=False):
        self.visualize_linear = linear

    def set_unsupervised(self, unsupervised=False):
        self.unsupervised = unsupervised

    def set_epochs(self, num_e):
        self.epochs = num_e if num_e > 0 else 1

    def train(self):
        print("Training")
        self._init_training()
        self._init_optimizer()

        # calculate steps per epoch for training and test set
        train_steps = len(self.training_dataset)
        test_steps = len(self.test_dataset)
        # initialize a dictionary to store training history
        self.Hist = {"train_loss": [], "test_loss": []}

        early_stopper = EarlyStopper(EARLY_STOP_PATIENCE)

        # loop over epochs
        print("[INFO] training the network...")
        self.log("[INFO] training the network...")
        self.log("[INFO] Learning rate is set to {}\n".format(self.learning_rate))
        startTime = time.time()
        for e in range(self.epochs):     #tqdm( ... for displaying)
            # set the model in training mode
            self.model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalTestLoss = 0
            # loop over the training set
            for (i, (x, y)) in enumerate(self.training_loader):
                # first, zero out any previously accumulated gradients, then
                self.optimizer.zero_grad()
                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                # red = torch.nn.functional.softmax(pred, dim=1)    # probablyTODO, check if it works better, nah
                # loss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                loss = self.loss_function[0](pred, y)       # only one loss function
                # perform backpropagation, and then update model parameters
                loss.backward()
                self.optimizer.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss

            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for (x, y) in self.test_loader:
                    # print(f"Shapes of input and output: {x.shape} | {y.shape}")
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred = self.model(x)
                    # test_loss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                    test_loss = self.loss_function[0](pred, y)
                    totalTestLoss += test_loss

                    if (e % 100) == 0 or e == self.epochs-1 or self.epochs < 50:  # random.randint(0, 100) > 90:
                        res = pred[0]
                        # res = torch.sigmoid(res)  # softmax is better; (run through sigmoid function < NO)
                        res = torch.nn.functional.softmax(res, dim=0)
                        # FurtherTODO maybe we want argmax, too? Please do it in all processing (train, execute and eval)
                        res = res.cpu().detach().numpy()  # squeeze(0). already done by converting
                        res = np.transpose(res, (1, 2, 0))  # H, W, C
                        mask_image = Ftrans.to_pil_image(res)
                        mask_image = self.postprocessing(mask_image)
                        my = y[0].cpu().detach().numpy()
                        my = np.transpose(my, (1,2,0))
                        orig_image = Ftrans.to_pil_image(my)
                        orig_image = self.postprocessing(orig_image)

                        orig_image.save("log/{}/{}_UNet_temp_res_{}_orig.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
                        mask_image.save("log/{}/{}_UNet_temp_res_{}_mask.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
            # adapt learning rate
            last_lr = self.learning_rate
            try:
                last_lr = self.scheduler.get_last_lr()    # ubuntu torch has a bug :(
            except Exception as lr_exception:
                print("[ERROR] torch problem, learning rate: {}".format(lr_exception))
            self.scheduler.step(totalTestLoss)

            # calculate the average training and validation loss
            avg_train_loss = totalTrainLoss / train_steps
            avg_test_loss = totalTestLoss / test_steps
            # update our training history
            self.Hist["train_loss"].append(avg_train_loss.item())
            self.Hist["test_loss"].append(avg_test_loss.item())
            # print(self.Hist)  # displays all up to now
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            print("Train loss: {:.8f}, Test loss: {:.6f}".format(avg_train_loss, avg_test_loss))
            self.log("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            self.log("Train loss: {:.8f}, Test loss: {:.6f}".format(avg_train_loss, avg_test_loss))
            try:
                if last_lr != self.scheduler.get_last_lr():
                   print("[INFO] learn.rate changes from {} to {}".format(last_lr, self.scheduler.get_last_lr()))
            except Exception as lr_exception2:
                print("[ERROR] learning rate scheduling: {}".format(lr_exception2))

            if early_stopper(totalTrainLoss):
                print(f"[INFO]: early stopping at epoch {e} with loss of {totalTrainLoss}")
                self.log(f"[INFO] Early stopping after {e} epochs with {totalTrainLoss}.")
                break

        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

        # perform model on any training data item to check its validity
        with torch.no_grad():
            self.model.eval()
            img, msk = self.training_dataset[0]
            img = img.unsqueeze(0).to(self.device)
            res = self.model(img)
            res = res.squeeze(0)
            # res = torch.sigmoid(res) # run through sidmoid function
            res = torch.nn.functional.softmax(res, dim=0)

        res = res.cpu().detach().numpy()        # squeeze(0). already done
        res = np.transpose(res, (1, 2, 0)) # H, W, C
        mask_image = Ftrans.to_pil_image(res)
        mask_image = self.postprocessing(mask_image)
        self.log(str(self.Hist))
        return mask_image, self.get_statistics_result(mask_image)

    def _get_transforms(self, data_augmentation=True):
        # the labels are already normalized
        if data_augmentation:
            tf = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                transforms.ToTensor(),  # includes normalizing to 0..1
                # transforms.Normalize((0, 0, 0), (255, 255, 255))  # transform from 0...256 to 0...1
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # transform from 0...1 to -1...1
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                transforms.ToTensor(),  # includes normalizing to 0..1
                # transforms.Normalize((0, 0, 0), (255, 255, 255))  # transform from 0...256 to 0...1
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # transform from 0...1 to -1...1
            ])
        return tf

    def _init_optimizer(self):
        print("[DEBUG] trying to set optimizer to " + self.optimizer_name)
        self.log("[DEBUG] trying to set optimizer to " + self.optimizer_name)

        if self.optimizer_name == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'AdaGrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'Adamax':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGDMomentum':
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)
        self.log(f"[DEBUG] optimizer is set to {self.optimizer.__class__.__name__}")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.7, patience=10)

    def _init_training(self):
        # retrieve paths (label files should be inside a folder called labels and the file is prefixed with labels)
        labels = [os.path.join(os.path.split(x)[0], "../labels/labels_"+os.path.split(x)[1]) for x in self.images]

        split = train_test_split(self.images, labels, test_size=0.2, random_state=42)
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        # f = open("./phybrosoft.log", "a+")
        self.log("Starting GMT {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
        self.log("\n".join(testImages))
        self.log(f"\nUsing UNet\n")
        #f.close()

        trans = self._get_transforms()
        self.training_dataset = SegmentationDataset(image_paths=trainImages, mask_paths=trainMasks, transforms=trans)
        self.test_dataset = SegmentationDataset(image_paths=testImages, mask_paths=testMasks, transforms=trans)

        if self.device == 'mps':
            self.training_loader = DataLoader(self.training_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                              pin_memory=True,
                                              num_workers=os.cpu_count(), multiprocessing_context="forkserver",
                                              persistent_workers=True)
            self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                          num_workers=os.cpu_count(), multiprocessing_context="forkserver",
                                          persistent_workers=True)
        else:
            self.training_loader = DataLoader(self.training_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                              num_workers=os.cpu_count())
            self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                              num_workers=os.cpu_count())

    def postprocessing(self, result, defaultToBackground=False):
        if self.classes == 1:
            return self.__postprocessing_direct(result)

        # return self.__postprocessing_threshold(result, defaultToBackground)
        if self.visualize_linear:
            return self.__postprocessing_rgb(result)
        else:
            return self.__postprocessing_best_class(result, defaultToBackground)

    def __postprocessing_best_class(self, result, defaultToBackground=False):
        pixels = result.getdata()
        for y in range(result.height):
            for x in range(result.width):
                cl = np.argmax(pixels[x + y * result.width])
                if cl == 0:     #if pixels[x + y * result.width][0] > THRESHOLD:  # bg
                    # #result.putpixel((x, y), (0, 0, 0))
                    # color = self.parameters[self.class_labels[0]+'_color']
                    color = self.parameters['background_color']
                elif cl == 1: #elif pixels[x + y * result.width][1] > THRESHOLD:  # healthy
                    #result.putpixel((x, y), (255, 0, 255))
                    #color = self.parameters[self.class_labels[2]+'_color']
                    color = self.parameters['healthy_color']
                elif cl == 2:       # elif pixels[x + y * result.width][2] > THRESHOLD:  # fibrosis
                    #result.putpixel((x, y), (255, 255, 0))
                    #color = self.parameters[self.class_labels[1]+'_color']
                    color = self.parameters['fibrosis_color']
                elif defaultToBackground:
                    # color = self.parameters[self.class_labels[0]+'_color']
                    color = self.parameters['background_color']
                else:
                    color = tuple(np.array(pixels[x+y*result.width]))
                result.putpixel((x,y), color)
        return result

    def __postprocessing_threshold(self, result, defaultToBackground=False):
        pixels = result.getdata()
        for y in range(result.height):
            for x in range(result.width):
                if pixels[x + y * result.width][0] > THRESHOLD:  # bg
                    # #result.putpixel((x, y), (0, 0, 0))
                    # color = self.parameters[self.class_labels[0]+'_color']
                    color = self.parameters['background_color']
                elif pixels[x + y * result.width][1] > THRESHOLD:  # healthy
                    #result.putpixel((x, y), (255, 0, 255))
                    #color = self.parameters[self.class_labels[2]+'_color']
                    color = self.parameters['healthy_color']
                elif pixels[x + y * result.width][2] > THRESHOLD:  # fibrosis
                    #result.putpixel((x, y), (255, 255, 0))
                    #color = self.parameters[self.class_labels[1]+'_color']
                    color = self.parameters['fibrosis_color']
                elif defaultToBackground:
                    # color = self.parameters[self.class_labels[0]+'_color']
                    color = self.parameters['background_color']
                else:
                    color = tuple(np.array(pixels[x+y*result.width]))
                result.putpixel((x,y), color)
        return result

    def __postprocessing_rgb(self, result): # tested method to generate results, can be disposed actually (just for lookup/reference)
        pixels = result.getdata()
        for y in range(result.height):
            for x in range(result.width):
                (r, g, b) = tuple(np.array(pixels[x + y * result.width]))       # FTrans scales up by 255 automatically
                if r > 255 or g > 255 or b > 255:
                    print(f"r, g, b - too large (>1.0) {r}, {g}, {b}")
                result.putpixel((x,y), (r, g, b))
        return result

    def __postprocessing_direct(self, result):
        pixels = result.getdata()
        ## DBG: # old_g = -1
        for y in range(result.height):
            for x in range(result.width):
                g = pixels[x + y * result.width]    # FTrans scales up by 255 automatically
                ## DBG: #if old_g != g:
                ## DBG: #    print( "{} {}".format(g, type(g)) )
                ## DBG: #    old_g = g
                if type(g) is tuple:
                    print("[DBG] found a tuple but expected")
                    result.putpixel((x,y), g)
                else:
                    result.putpixel((x, y), int(g))
        return result

    def _creating_execution_dataloader(self, imagefile):
        trans = self._get_transforms(False)
        dataset = SegmentationDataset([imagefile], [imagefile], transforms=trans)
        if self.device == 'mps':
            return DataLoader(dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1,
                              multiprocessing_context="forkserver", persistent_workers=False)
        return DataLoader(dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    def execute(self, imagefile):
        # image = Image.open(imagefile)
        # if we wouldn't use the transformations, we'd have W x H x C, then we would need to prepare for
        #    B x C x H x W
        # image = np.transpose(image, (2, 0, 1))
        # image = np.expand_dims(image, 0)
        # but we transform it like this
        #imagetensor = trans(image)
        #imagetensor.unsqueeze(0)        # adding a batch index/dimension   (now it is done by dataloader)
        loader = self._creating_execution_dataloader(imagefile)
        for (i, (image)) in loader:
            imagetensor = image.to(self.device)
            self.model.eval()
            with torch.no_grad():
                result = self.model(imagetensor)
                result = result.squeeze(0)
                # result = torch.sigmoid(result) #we are using softmax, as we create multi-segmentation masks (exclusive)
                result = torch.nn.functional.softmax(result, dim=0)
            res = result.cpu().detach().numpy()
            res = np.transpose(res,(1,2,0))
        mask_image = self.postprocessing(Ftrans.to_pil_image(res),True)
        return mask_image, self.get_statistics_result(mask_image)

    def get_statistics_result(self, mask_image=None):
        class_counter = [1] * len(self.class_labels)
        if mask_image is not None:
            # the 1 is for the laplacian correction, in order to avoid division by 0 errors
            width, height = mask_image.size
            imgdata = mask_image.getdata()
            for y in range(0, height):
                for x in range(0, width):
                    rgb = imgdata[y*width+x]
                    for check_idx in range(len(self.class_labels)):
                        if rgb == self.parameters[self.class_labels[check_idx] + '_color']:
                            class_counter[check_idx] += 1

        return mlprocessing.StatisticsResult(self.class_labels, class_counter)

    def save_model(self, filename):
        torch.save(self.model, filename)

    def load_model(self, filename):
        try:
            self.model = torch.load(filename, weights_only=False)
        except RuntimeError as err:
            try:
                self.model = torch.load(filename, weights_only=False, map_location=torch.device(self.device))
            except RuntimeError as err2:
                raise mlexceptions.MLFileTypeException("Cannot load this model as Unet, probably wrong file format?")
