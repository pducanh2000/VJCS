import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader.datasets import *
from losses.loss_selector import Loss_Selector
from utils.utils import data_split, export_data
from models.models import Model


def train_loso(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion = None
    max_test_accs = []
    model_cfg = cfg["models"][args["model"]]

    for subject_out in range(1, 14):
        print(f"################ Subject {subject_out} #################")
        train_data, test_data = data_split('S' + str(subject_out))

        # Dataset
        if args["loss"] == "contrastive":
            train_dataset = PostureDatasetContrastive(train_data, model_cfg)
            test_dataset = PostureDatasetContrastive(test_data, model_cfg)
        elif args["loss"] == "triplet":
            train_dataset = PostureDatasetTriplet(train_data, model_cfg)
            test_dataset = PostureDatasetTriplet(test_data, model_cfg)
        else:
            train_dataset = PostureDataset(train_data, model_cfg)
            test_dataset = PostureDataset(test_data, model_cfg)

        # Data Loader
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

        # Model
        model = Model(model_cfg).to(device)

        # Define Losses
        criterion1 = CrossEntropyLoss().to(device)
        loss_parameters = cfg["losses"][args["loss"]]
        if args["losses"] in ["a_softmax", "am_softmax", "center", "contrast_center"]:
            loss_parameters["feat_dim"] = model_cfg["feat"]
            loss_parameters["num_classes"] = 17

        loss_selector = Loss_Selector()
        criterion2 = loss_selector(args["loss"], **loss_parameters).to(device)

        # Optimizer
        if args["losses"] in ["a_softmax", "am_softmax", "center", "contrast_center"]:
            optimizer = Adam([
                            {'model_params': model.parameters()},
                            {'loss_params': criterion2.parameters()}
                            ], lr=args["lr"], weight_decay=args["l2_regularization"])
        else:
            optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args["l2_regularization"])

        scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.95)
        max_test_accs.append(0)

        for epoch in range(args["epoch_n"]):
            print('***** Epoch: *****', epoch + 1)
            model.train()
            train_epoch_iterator = tqdm(train_loader,
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


