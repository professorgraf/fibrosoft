
import os
import random
import time

import numpy as np
from PIL import Image

from machinelearning import mlprocessing
from machinelearning import mlexceptions

import torch
import torch.nn as nn
from torchvision.transforms import Grayscale
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as Ftrans
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from machinelearning.unet import (Encoder, Decoder, MulticlassDiceLoss, NoLoss, UNetProcessing, EarlyStopper,
                                  SegmentationDataset, UnsupervisedMaskedTrainingDataset,
                                  EARLY_STOP_PATIENCE, BATCH_SIZE, INIT_LR, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT,
                                  NUM_EPOCHS, ENC_CHANNELS, DEC_CHANNELS)

# necessary modifications for 1 byte per pixel / 1 channel
LOSS_FCT = ['CE', 'DICE']   # ('DICE', 'CE')    # 'CE' / 'BCE' / 'DICE' / 'L1' / 'CE_W' ... or no loss
ENC_CHANNELS_1CHAN = (1,) + (ENC_CHANNELS[1:])
DEC_CHANNELS_1CHAN = DEC_CHANNELS


class UNet1by1(nn.Module):
    # UNET class constructor
    def __init__(self, enc_channels=(1, 16, 32, 64), dec_channels=(64, 32, 16),
                 nb_classes=3, retain_dim=True,
                 out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)):

        super().__init__()
        # initialize the encoder and decoder
        self.convolve_to_1channel = nn.Conv2d(3, 1, 1)            # 1by1 convolution
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(dec_channels[-1], nb_classes, 1)
        self.retainDim = retain_dim
        self.outSize = out_size

    def forward(self, x):
        # pre 1by1 convolution
        xx = self.convolve_to_1channel(x)
        # grab the features from the encoder
        enc_features = self.encoder(xx)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[::-1][0],
            enc_features[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        segmentation_map = self.head(dec_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            segmentation_map = F.interpolate(segmentation_map, self.outSize)
        # return the segmentation map
        return segmentation_map


class UNetGray(nn.Module):
    # UNET class constructor
    def __init__(self, enc_channels=(1, 16, 32, 64), dec_channels=(64, 32, 16),
                 nb_classes=3, retain_dim=True,
                 out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)):

        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(dec_channels[-1], nb_classes, 1)
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
        segmentation_map = self.head(dec_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            segmentation_map = F.interpolate(segmentation_map, self.outSize)
        # return the segmentation map
        return segmentation_map


class UNet1by1Processing(UNetProcessing):
    def __init__(self, classes=3, epochs=NUM_EPOCHS, learning_rate=INIT_LR, optimizer='SGD', train_on_gpu=True,
                 visualize_linear=False, unsupervised=False):
        super().__init__(classes, epochs, learning_rate, optimizer, train_on_gpu, visualize_linear, unsupervised)
        # self.training_dataset = None
        #self.test_dataset = None
        #self.model = None
        #self.train_on_gpu = train_on_gpu
        #self.epochs = epochs
        #self.learning_rate = learning_rate
        #self.training_loader = None
        #self.test_loader = None
        #self.optimizer_name = optimizer
        #self.Hist = {"train_loss": [0], "test_loss": [0]}

        #self.device = "cpu"
        #if self.train_on_gpu and torch.cuda.is_available():
        #    self.device = "cuda:0"
        #elif self.train_on_gpu and torch.backends.mps.is_available():
        #    self.device = "mps"
        #print(f"[INFO] training on {self.device}")

        self.model = UNet1by1(enc_channels=ENC_CHANNELS_1CHAN, dec_channels=DEC_CHANNELS_1CHAN,
                            nb_classes=classes, retain_dim=True, out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        self.model.to(self.device)

        #self.loss_function = [None, None]
        #for i in range(len(LOSS_FCT)):
        #    if LOSS_FCT[i] == 'BCE':
        #        self.loss_function[i] = torch.nn.BCEWithLogitsLoss()
        #    elif LOSS_FCT[i] == 'CE':
        #        self.loss_function[i] = torch.nn.CrossEntropyLoss()
        #    elif LOSS_FCT[i] == "DICE":
        #        self.loss_function[i] = MulticlassDiceLoss(num_classes=3, softmax_dim=1)
        #    elif LOSS_FCT[i] == 'NLL':
        #        self.loss_function[i] = torch.nn.NLLLoss()
        #    elif LOSS_FCT[i] == 'L1':
        #        self.loss_function[i] = torch.nn.L1Loss()
        #    else:
        #        self.loss_function[i] = NoLoss()    # skip a loss

        #self.optimizer = None
        #self.scheduler = None

    def train(self):
        print("Training")
        self._init_training()
        self._init_optimizer()

        early_stopper = EarlyStopper(EARLY_STOP_PATIENCE)

        # calculate steps per epoch for training and test set
        trainSteps = len(self.training_dataset)
        testSteps = len(self.test_dataset)
        # initialize a dictionary to store training history
        self.Hist = {"train_loss": [], "test_loss": []}

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
                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                self.optimizer.zero_grad()
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
                    testloss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                    totalTestLoss += testloss

                    if (e % 100) == 0 or e == self.epochs-1 or self.epochs < 50:  # random.randint(0, 100) > 90:
                        res = pred[0]
                        # res = torch.sigmoid(res) # run through sigmoid function
                        res = torch.nn.functional.softmax(res, dim=0)  # or run through softmax (non different.)
                        res = res.cpu().detach().numpy()  # squeeze(0). already done
                        res = np.transpose(res, (1, 2, 0))  # H, W, C
                        mask_image = Ftrans.to_pil_image(res)
                        mask_image = self.postprocessing(mask_image)
                        my = y[0].cpu().detach().numpy()
                        my = np.transpose(my, (1,2,0))
                        orig_image = Ftrans.to_pil_image(my)
                        orig_image = self.postprocessing(orig_image)

                        orig_image.save("log/{}/{}_UNet1by1_temp_{}_orig.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
                        mask_image.save("log/{}/{}_UNet1by1_temp_{}_mask.png".format(
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
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps
            # update our training history
            #self.Hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            #self.Hist["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            self.Hist["train_loss"].append(avgTrainLoss.item())
            self.Hist["test_loss"].append(avgTestLoss.item())
            # print(self.Hist)  # displays all up to now
              # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            print("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
            self.log("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            self.log("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
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

    def _init_training(self):
        # retrieve paths (label files should be inside a folder called labels and the file is prefixed with labels)
        labels = [os.path.join(os.path.split(x)[0], "../labels/labels_"+os.path.split(x)[1]) for x in self.images]

        split = train_test_split(self.images, labels, test_size=0.2, random_state=42)
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        self.log("Starting GMT {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
        self.log("\n".join(testImages))
        self.log(f"\nUsing {self.model.__class__.__name__}\n")

        trans = self._get_transforms()

        if not self.unsupervised:
            self.training_dataset = SegmentationDataset(image_paths=trainImages, mask_paths=trainMasks,
                                                        transforms=trans)
            self.test_dataset = SegmentationDataset(image_paths=testImages, mask_paths=testMasks, transforms=trans)
        else:
            self.training_dataset = UnsupervisedMaskedTrainingDataset(image_paths=trainImages, transforms=trans,
                                                                      neighborhood=1)
            self.test_dataset = UnsupervisedMaskedTrainingDataset(image_paths=testImages, transforms=trans,
                                                                  neighborhood=1)

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

    def execute(self, imagefile):
        # image = Image.open(imagefile) - when opening, we use now
        # if we wouldn't use the transformations/the data loaders, we'd have W x H x C, then we would need to prepare for
        #    B x C x H x W
        # image = np.transpose(image, (2, 0, 1))
        # image = np.expand_dims(image, 0)
        # but we transform it like this
        #imagetensor = trans(image)
        #imagetensor.unsqueeze(0)        # adding a batch index/dimension   (now it is done by dataloader)
        loader = self._creating_execution_dataloader(imagefile)
        #    needs to be B x C x H x W (dataloader takes care of it)
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


class UNetGrayProcessing(UNet1by1Processing):
    def __init__(self, classes=3, epochs=NUM_EPOCHS, learning_rate=INIT_LR, optimizer='SGD', train_on_gpu=True,
                 visualize_linear=False, unsupervised=False):
        super().__init__(classes, epochs, learning_rate, optimizer, train_on_gpu, visualize_linear, unsupervised)
        self.model = UNetGray(enc_channels=ENC_CHANNELS_1CHAN, dec_channels=DEC_CHANNELS_1CHAN,
                              nb_classes=classes, retain_dim=True, out_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        self.model.to(self.device)

    def train(self):
        print("Training")
        self._init_training()
        self._init_optimizer()

        early_stopper = EarlyStopper(EARLY_STOP_PATIENCE)

        # calculate steps per epoch for training and test set
        trainSteps = len(self.training_dataset)
        testSteps = len(self.test_dataset)
        # initialize a dictionary to store training history
        self.Hist = {"train_loss": [], "test_loss": []}

        # loop over epochs
        print("[INFO] training the network...")
        self.log("[INFO] training the network...")
        self.log("[INFO] Learning rate is set to {}\n".format(self.learning_rate))
        startTime = time.time()

        grayscaler = Grayscale(num_output_channels=1)
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
                # print(f"Shapes of input and output: {x.shape} | {y.shape}")
                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                x = grayscaler(x)
                x.unsqueeze(1)
                # print(f"Shapes of input after expanding: {x.shape}")
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                # perform backpropagation, and then update model parameters
                loss.backward()
                self.optimizer.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss
            # switch off autograd - do evaluation
            with ((((torch.no_grad())))):
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for (x, y) in self.test_loader:
                    # print(f"Shapes of input and output: {x.shape} | {y.shape}")
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    x = grayscaler(x)
                    x.unsqueeze(1)
                    # make the predictions and calculate the validation loss
                    pred = self.model(x)
                    testloss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                    totalTestLoss += testloss

                    if (e % 100) == 0 or e == self.epochs-1 or self.epochs < 50:  # random.randint(0, 100) > 90:
                        res = pred[0]
                        # res = torch.sigmoid(res) # run through sigmoid function
                        res = torch.nn.functional.softmax(res, dim=0)  # or run through softmax (non different.)
                        res = res.cpu().detach().numpy()  # squeeze(0). already done
                        res = np.transpose(res, (1, 2, 0))  # H, W, C
                        mask_image = Ftrans.to_pil_image(res)
                        mask_image = self.postprocessing(mask_image)
                        my = y[0].cpu().detach().numpy()
                        my = np.transpose(my, (1,2,0))
                        orig_image = Ftrans.to_pil_image(my)
                        orig_image = self.postprocessing(orig_image)

                        orig_image.save("log/{}/{}_UNetGray_temp_{}_orig.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
                        mask_image.save("log/{}/{}_UNetGray_temp_{}_mask.png".format(
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
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps
            # update our training history
            #self.Hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            #self.Hist["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            self.Hist["train_loss"].append(avgTrainLoss.item())
            self.Hist["test_loss"].append(avgTestLoss.item())
            # print(self.Hist)  # displays all up to now
              # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            print("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
            self.log("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            self.log("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
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
            img = grayscaler(img)
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

    def execute(self, imagefile):
        loader = self._creating_execution_dataloader(imagefile)
        #    needs to be B x C x H x W (dataloader takes care of it)
        grayscaler = Grayscale(num_output_channels=1)
        for (i, (image)) in loader:
            imagetensor = image.to(self.device)
            imagetensor = grayscaler(imagetensor)
            imagetensor.unsqueeze(1)    # channels
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
