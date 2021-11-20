import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.datasets import *
from losses.loss_selector import Loss_Selector
from utils.utils import data_split, export_data
from models.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


class Engine(object):
    def __init__(self, args, train_loader, test_loader, model, optimizer, scheduler, criterion1, criterion2, device):
        self.loss_name = args["loss"]
        self.model = model
        self.optimizer = optimizer
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.ratio1 = args['ratio1']
        self.ratio2 = args['ratio2']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.scheduler = scheduler

    def train(self):
        self.model.train()
        train_epoch_iterator = tqdm(self.train_loader,
                                    desc="Training (Step X) (loss=X.X)",
                                    bar_format="{l_bar}{r_bar}",
                                    dynamic_ncols=True, )
        train_losses = []
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        train_acc_scores = []
        predicts = None
        labels = None

        for id, (images, postures) in enumerate(train_epoch_iterator):
            images = images.to(self.device)
            postures = postures.to(self.device)
            feature, predict = self.model(images)
            loss = self.ratio1 * self.criterion1(predict, postures) + self.ratio2 * self.criterion2(feature, postures)
            train_epoch_iterator.set_description(
                "Training (Step %d) (loss=%2.5f)" % (id + 1, loss.item())
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())

            _, postures_pred = torch.max(predict, dim=1)

            if predicts is None:
                predicts = np.array(postures_pred.cpu())
            else:
                predicts = np.concatenate((predicts, postures_pred.cpu()))

            if labels is None:
                labels = np.array(postures.cpu())
            else:
                labels = np.concatenate((labels, postures.cpu()))
            train_precisions.append(
                precision_score(postures_pred.cpu(), postures.cpu(), average='macro', zero_division=0))
            train_recalls.append(
                recall_score(postures_pred.cpu(), postures.cpu(), average='macro', zero_division=0))
            train_f1_scores.append(f1_score(postures_pred.cpu(), postures.cpu(), average='macro', zero_division=0))
            train_acc_scores.append(accuracy_score(postures_pred.cpu(), postures.cpu()))

        print('Train loss: ', sum(train_losses) / len(train_losses))
        print('Train score: Precision = %2.5f, Recall = %2.5f, F1_score = %2.5f, Acc_score = %2.5f' % (
            sum(train_precisions) / len(train_precisions),
            sum(train_recalls) / len(train_recalls), sum(train_f1_scores) / len(train_f1_scores),
            sum(train_acc_scores) / len(train_acc_scores)))
        self.scheduler.step()

    def test(self):
        self.model.eval()

    def save_checkpoint(self):
        pass