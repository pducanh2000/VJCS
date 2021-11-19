import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor, Normalize

import numpy as np
import cv2


class PostureDataset(Dataset):
    '''Create sleeping posture dataset
    Arguments
    ----------
    data: dict
        Dictionary includes image paths, posture types

    preprocess: bool
        Determine this data use preprocessing or not 

    Returns
    ----------
    torch.Tensor
        Tensor: (image, posture)
    '''
    def __init__(self, data, preprocess=True):
        super(PostureDataset, self).__init__()
        self.images = data["images"]
        self.postures = data["postures"].reshape(-1)
        self.preprocessing = preprocess

    def __getitem__(self, item):
        if self.preprocessing is True:
            data_item = self.preprocess(self.images[item], self.postures[item])
        else:
            data_item = self.transform(self.images[item], self.postures[item])
        
        return data_item

    def __len__(self):

        return len(self.postures)

    @staticmethod
    def preprocess(image, posture):
        image = cv2.equalizeHist(image)
        
        image = ToPILImage()(image)
        image = image.convert('RGB')
        image = image.resize((112, 224))

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image, torch.tensor(posture, dtype=torch.long)

    @staticmethod
    def transform(image, posture):
        image = ToPILImage()(image)
        image = image.convert('RGB')

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image, torch.tensor(posture, dtype=torch.long)



class PostureContrastiveDataset(Dataset):
    '''Create sleeping posture dataset for training with contrastive loss
    
    Arguments
    ----------
    data: dict
        Dictionary includes image paths, posture types

    train_mode: bool
        train mode or eval mode?

    preprocess: bool
        Determine this data use preprocessing or not 

    Returns
    ----------
    torch.Tensor:
        tensor (image1, image2, similar_label, posture_labe)
    '''
    def __init__(self, data, train_mode=True, preprocess=True):
        super(PostureContrastiveDataset, self).__init__()
        self.train_mode = train_mode
        self.images = data["images"]
        self.postures = data["postures"].reshape(-1)
        
        
        self.postures_set = set(self.postures)
        self.label_to_indices = {label: np.where(self.postures == label)[0] for label in self.postures_set}

        self.preprocessing = preprocess
        
        if self.train_mode == False:
            random_state = np.random.RandomState(2000)
            
            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.postures[i]]), 1, self.postures[i]] 
                            for i in range(0, len(self.postures), 2)]
                                
            negative_pairs = [[i, random_state.choice(self.label_to_indices[
                            np.random.choice(list(self.postures_set - set([self.postures[i]])))]), 0, self.postures[i]] 
                            for i in range(1, len(self.postures), 2)]
            
            self.test_pairs = positive_pairs + negative_pairs
            
    def __getitem__(self, item):
        if self.train_mode:
            target = np.random.randint(0, 2)
            img1, post1 = self.images[item], self.postures[item]
            if target == 1:
                siamese_item = item
                while siamese_item == item:
                    siamese_item = np.random.choice(self.label_to_indices[post1])
            else:
                siamese_post = np.random.choice(list(self.postures_set - set([post1])))
                siamese_item = np.random.choice(self.label_to_indices[siamese_post])
            img2 = self.images[siamese_item]
        else:
            img1 = self.images[self.test_pairs[item][0]]
            img2 = self.images[self.test_pairs[item][1]]
            target = self.test_pairs[item][2]
            post1 = self.test_pairs[item][3]
            
        
        if self.preprocessing is True:
            img1 = self.preprocess(img1)
            img2 = self.preprocess(img2)
        else:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, target, post1

    def __len__(self):

        return len(self.postures)

    @staticmethod
    def preprocess(image):
        image = cv2.equalizeHist(image)
        if np.random.rand(0, 1) < 0.5:
            image = cutout(image, n_holes=5, length=4)
        
        image = ToPILImage()(image)
        image = image.convert('RGB')
        image = image.resize((112, 224))

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image

    @staticmethod
    def transform(image):

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image
