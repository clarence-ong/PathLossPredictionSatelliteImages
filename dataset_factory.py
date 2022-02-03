
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from skimage import io, transform
import os
import matplotlib.pyplot as plt

def dataset_factory(use_images=True, 
height_folder="/content/PathLossPredictionSatelliteImages/Data_Folder/Height_Images_2_resized",
Tx_Azimuth_Folder = "/content/PathLossPredictionSatelliteImages/Data_Folder/Tx_Azimuth_2_resized",
Tx_Elevation_Folder = "/content/PathLossPredictionSatelliteImages/Data_Folder/Tx_Elevation_Angle_2_resized",
Rx_Azimuth_Folder = "/content/PathLossPredictionSatelliteImages/Data_Folder/Rx_Azimuth_2_resized",
Rx_Elevation_Folder = "/content/PathLossPredictionSatelliteImages/Data_Folder/Tx_Elevation_Angle_2_resized",
transform=True, data_augment_angle=10):
    #Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI_64,PCI_65,PCI_302	
    #selected_features = [0, 1, 3, 4, 5, 6, 7, 8]
    #['Tx_Lon', 'Tx_Lat', 'Rx_Lon', 'Rx_Lat', 'Tx_Height', 'Rx_Height','Tx_Rx_Distance']
    selected_features = [0, 1, 2, 3, 4, 5, 6]
    # ['SINR', 'RSRP', 'RSRQ', 'Power']	
    #selected_targets = [1]
    # ["RSS"]
    selected_targets = [0]
    print(selected_features)
    dataset_path='/content/PathLossPredictionSatelliteImages/Data_Folder' 
    features = np.load("{}/training_features.npy".format(dataset_path))
    targets = np.load("{}/training_targets.npy".format(dataset_path))
    test_features =  np.load("{}/test_features.npy".format(dataset_path))
    test_targets = np.load("{}/test_targets.npy".format(dataset_path))    
    target_mu = np.load("{}/targets_mu.npy".format(dataset_path))    
    target_std = np.load("{}/targets_std.npy".format(dataset_path))    
    features_mu = np.load("{}/features_mu.npy".format(dataset_path))    
    features_std = np.load("{}/features_std.npy".format(dataset_path))
    images = np.load("{}/train_image_idx.npy".format(dataset_path))
    test_images = np.load("{}/test_image_idx.npy".format(dataset_path))

    

    features = features[:, selected_features]
    test_features = test_features[:, selected_features]
    features_mu = features_mu[selected_features]
    features_std = features_std[selected_features]


    targets = targets[:, selected_targets]
    test_targets = test_targets[:, selected_targets]
    target_mu = target_mu[selected_targets]
    target_std = target_std[selected_targets]



    # Data augmentation
    if transform:
        #composed = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.RandomAffine(data_augment_angle, shear=10), transforms.ToTensor()])
        composed = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()])
    else:
        composed = None
    
    # Dataset
    train_dataset = DrivetestDataset(features, targets, images, target_mu, target_std, features_mu, features_std, use_images, height_folder, Tx_Azimuth_Folder, Tx_Elevation_Folder, Rx_Azimuth_Folder, Rx_Elevation_Folder, transform=composed)
    #valid_dataset = DrivetestDataset(images, features, targets, valid_idx, target_mu, target_std, features_mean, features_std, use_images, image_folder)
    test_dataset = DrivetestDataset(test_features, test_targets, test_images,  target_mu, target_std, features_mu, features_std, use_images, height_folder, Tx_Azimuth_Folder, Tx_Elevation_Folder, Rx_Azimuth_Folder, Rx_Elevation_Folder, transform=transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()]))
    return train_dataset, test_dataset


class DrivetestDataset(Dataset):
    def __init__(self, features, targets, images, target_mu, target_std, feature_mu, feature_std, use_images, height_folder, Tx_Azimuth_Folder, Tx_Elevation_Folder, Rx_Azimuth_Folder, Rx_Elevation_Folder, transform=None):
        self.features = features
        self.targets = targets
        self.image_idx = images
        self.target_mu = target_mu
        self.target_std = target_std
        self.feature_mu = feature_mu
        self.feature_std = feature_std
        self.distances = (self.features[:,2] * self.feature_std[2])+self.feature_mu[2]
        self.targets_unnorm = (self.targets * self.target_std)+self.target_mu
        self.use_images = use_images
        self.height_folder = height_folder
        self.Tx_Azimuth_Folder = Tx_Azimuth_Folder
        self.Tx_Elevation_Folder = Tx_Elevation_Folder
        self.Rx_Azimuth_Folder = Rx_Azimuth_Folder
        self.Rx_Elevation_Folder = Rx_Elevation_Folder
        self.transform = transform

    def get_811Mhz_idx(self):
        return np.argwhere(np.asarray(self.features[:,7] != 1))

    def get_2630Mhz_idx(self):
        return np.argwhere(np.asarray(self.features[:,7] == 1))

    def __getitem__(self, index):
        idx = self.image_idx[index]
        X = torch.from_numpy(self.features[index]).float() # Features (normalized)
        if self.use_images:
            if self.height_folder == None: #images are then pointer to hdf5
                image = self.image_idx[index]
            else:
                height_name = os.path.join(self.height_folder, "{}.jpg".format(idx))
                height_image = io.imread(height_name)
                height_image = height_image / 255

                Tx_A_name = os.path.join(self.Tx_Azimuth_Folder, "{}.jpg".format(idx))
                Tx_A_image = io.imread(Tx_A_name)
                Tx_A_image = Tx_A_image / 255

                Tx_E_name = os.path.join(self.Tx_Elevation_Folder, "{}.jpg".format(idx))
                Tx_E_image = io.imread(Tx_E_name)
                Tx_E_image = Tx_E_image / 255

                Rx_A_name = os.path.join(self.Rx_Azimuth_Folder, "{}.jpg".format(idx))
                Rx_A_image = io.imread(Rx_A_name)
                Rx_A_image = Rx_A_image / 255

                Rx_E_name = os.path.join(self.Rx_Elevation_Folder, "{}.jpg".format(idx))
                Rx_E_image = io.imread(Rx_E_name)
                Rx_E_image = Rx_E_image / 255

            A = torch.from_numpy(height_image).float().permute(2,0,1)
            B = torch.from_numpy(Tx_A_image).float().permute(2,0,1)
            C = torch.from_numpy(Tx_E_image).float().permute(2,0,1)
            D = torch.from_numpy(Rx_A_image).float().permute(2,0,1)
            E = torch.from_numpy(Rx_E_image).float().permute(2,0,1)
            
        else:
            A = torch.tensor(0)
            B = torch.tensor(0)
            C = torch.tensor(0)
            D = torch.tensor(0)
            E = torch.tensor(0)
        y = torch.from_numpy(self.targets[index]).float() # Target
        dist = torch.abs(torch.tensor(self.distances[index])).float().view(1) # Unormalized distance
        dist = dist * 1000 # to meters

        if self.use_images:
            if self.transform:
                A = self.transform(A)
                B = self.transform(B)
                C = self.transform(C)
                D = self.transform(D)
                E = self.transform(E)

                Composed_Image = torch.cat((A, B, C, D, E), 0)

        return X, Composed_Image, y, dist

    def __len__(self):
        return len(self.features)

if __name__ == '__main__':
    train, test = dataset_factory()
    data = train.__getitem__(2)
    print(data[1].shape)