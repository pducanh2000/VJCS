def train(yaml_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion = None
    max_test_accs = []
    
    for subject_out in range(1, 14):
        print(f"################ Subject {subject_out} #################")
        train_data, test_data = data_split(yaml_data, 'S' + str(subject_out))

        train_dataset = PostureDataset(train_data)
        test_dataset = PostureDataset(test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=yaml_data["train_batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=yaml_data["test_batch_size"], shuffle=True)
        
        # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=17).to(device)
        model = get_model(model_name).to(device)

        criterion = TripletLoss(margin=1.0).to(device)
        ce = nn.CrossEntropyLoss().to(device)
        optimizer = Adam(model.parameters(), lr=yaml_data["lr"], weight_decay=yaml_data["l2_regularization"])
        scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.95)
        max_test_accs.append(0)
        # tree = XGBClassifier(seed = 2021)
      
        for epoch in range(yaml_data["num_epochs"]):
            print('***** Epoch: *****', epoch + 1)
            model.train()
            train_epoch_iterator = tqdm(train_loader,
                                        desc="Training (Step X) (loss=X.X)",
                                        bar_format="{l_bar}{r_bar}",
                                        dynamic_ncols=True,)
            train_losses = []
            predicts = None
            labels = None

            for id, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_epoch_iterator):
                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                negative_img = negative_img.to(device)
                
                optimizer.zero_grad()
                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)
                if model_name == 'inception_v3':
                    anchor_out = anchor_out[0]
                    positive_out = positive_out[0]
                    negative_out = negative_out[0]
                loss1 = criterion(anchor_out, positive_out, negative_out)
                loss2 = ce(anchor_out,anchor_label.to(device))
                loss = loss2*0.9+loss1*0.1

                train_epoch_iterator.set_description(
                    "Training (Step %d) (loss=%2.5f)" % (id + 1, loss.item())
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                          
            # print('Train loss: ', sum(train_losses) / len(train_losses))
            scheduler.step()            

            # tree.fit(predicts,labels)

            test_losses = []
            predicts = None
            labels = None
            model.eval()
            test_epoch_iterator = tqdm(test_loader,
                                        desc="Training (Step X) (loss=X.X)",
                                        bar_format="{l_bar}{r_bar}",
                                        dynamic_ncols=True,)
            with torch.no_grad():
              for id, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(test_epoch_iterator):
                  anchor_img = anchor_img.to(device)
                  anchor_out = model(anchor_img)
                  if predicts is None:
                      predicts = np.array(anchor_out.cpu())
                  else:
                      predicts = np.concatenate((predicts, anchor_out.cpu()))
                  if labels is None:
                      labels = np.array(anchor_label.cpu())
                  else:
                      labels = np.concatenate((labels, anchor_label.cpu()))

            # predict_ = tree.predict(predicts)
            predicts = np.argmax(predicts,axis=1)

            test_precisions=[precision_score(predicts, labels, average='macro')]
            test_recalls=[recall_score(predicts, labels, average='macro')]
            test_f1_scores=[f1_score(predicts, labels, average='macro')]
            test_acc_scores=[accuracy_score(predicts, labels)]

            print('Test score: Precision = %2.5f, Recall = %2.5f, F1_score = %2.5f, Acc_score = %2.5f' % (sum(test_precisions) / len(test_precisions),\
                sum(test_recalls) / len(test_recalls), sum(test_f1_scores) / len(test_f1_scores), sum(test_acc_scores) / len(test_acc_scores)))
            
            if max_test_accs[-1] < sum(test_acc_scores) / len(test_acc_scores):
                max_test_accs[-1] = sum(test_acc_scores) / len(test_acc_scores)
                if max_test_accs[-1] > 0.99:
                  break
            print("Max accuracy score", max_test_accs[-1])
       

    print("Max test accuracy: ", max_test_accs)
    print('Accuracy: ', sum(max_test_accs) / len(max_test_accs))