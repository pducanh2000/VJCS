import torch
from torch.utils.data import Dataset


class PostureDataset(Dataset):
    def __init__(self, data, model_cfg, use_preprocess=True):
        super(PostureDataset, self).__init__()
        self.images = data["images"]
        self.postures = data["postures"].reshape(-1)
        self.preprocessing = use_preprocess
        self.index = np.array((range(len(self.postures))))
        self.model_cfg = model_cfg

    def __getitem__(self, item):
        anchor_img = self.images[item]
        anchor_label = self.postures[item]
        positive_list = self.index[self.index!=item][self.postures[self.index!=item]==anchor_label]
        positive_item = random.choice(positive_list)
        positive_img = self.images[positive_item]
        negative_list = self.index[self.index!=item][self.postures[self.index!=item]!=anchor_label]
        negative_item = random.choice(negative_list)
        negative_img = self.images[negative_item]
        if self.preprocessing:
            anchor_img = preprocess(anchor_img)
            positive_img = preprocess(positive_img)
            negative_img = preprocess(negative_img)
        else:
            anchor_img = self.transform(anchor_img)
        return anchor_img, positive_img, negative_img, torch.tensor(anchor_label, dtype=torch.long)

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