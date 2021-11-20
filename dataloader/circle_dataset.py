import torch
from torch.utils.data import Dataset


class PostureDataset(Dataset):
    def __init__(self, data,model_cfg,preprocess=True):
        super(PostureDataset, self).__init__()
        self.images = data["images"]
        self.postures = data["postures"].reshape(-1)
        self.preprocessing = preprocess
        self.model_cfg = model_cfg

    def __getitem__(self, item):
        if self.preprocessing is True:
            data_item = self.preprocess(self.images[item], self.postures[item])
        else:
            data_item = self.transform(self.images[item], self.postures[item])
        
        return data_item

    def __len__(self):
        return len(self.postures)

    @staticmethod
    def preprocess(image):
        image = cv2.equalizeHist(image)
        
        image = ToPILImage()(image)
        image = image.convert('RGB')
        image = image.resize(self.model_cfg['size'])

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image

