import torch
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.autograd import Variable
import numpy as np
import time
import gc

def train_bs(dataloader_train, dataloader_dev, model, hyper_params, device):
    model = model.cuda()
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters in network = {}'. format(total_num_params))
    print('Total number of trainable  parameters in network = {}'. format(total_num_trainable_params))
    #Loss criterion
    criterion = torch.nn.BCELoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss()

    if hyper_params.get('optimizer') == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get('learning_rate'))
        #optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get('learning_rate'),weight_decay=1e-1)
        print("Adam optimizer set")
    else:
        print("Please choose proper optimizer for training")

    profile = open("../meta/profile.time", "w")
    profile_dev = open("../meta/profile_dev.time", "w")
    date = datetime.now()

    training_loss_list = []
    dev_loss_list = []
    for epoch in range(hyper_params['epochs']):
        epochStartTime = time.time()
        running_loss = 0.0
        running_dev_loss = 0.0
        #n_batches = len(train_loader)
        batch_size = hyper_params['bs']
        n_batches = len(dataloader_train)
        print("range of dataloader ", n_batches)
        model.train()
        for i, batch in enumerate(dataloader_train):
            #print("INSIDE enumerate")
            t1 = time.time()
            inputs = torch.reshape(batch[0], (-1, 360)).cuda()
            #inputs = torch.reshape(batch[0], (-1, 440)).cuda()
            labels = torch.reshape(batch[1], (-1, 42)).cuda()
            if i == 0: print(inputs.shape, inputs.device, labels.shape, labels.device)

            inputs = inputs.requires_grad_()

            # criterion.to(device)

            # Initializing model gradients to zero
            optimizer.zero_grad()

            # Data feed-forward through the network
            outputs = model(inputs)
            # Predicted class is the one with maximum probability
            # Finding the MSE
            loss = criterion(outputs, labels)
            loss.backward()

            # Update the network parameters
            optimizer.step()

            # Accumulate loss per batch
            running_loss += loss.item()
            if i % 500 == 0:
                #torch.save(model.state_dict(),'model_epoch_{}_loss_{}.pt'.format(epoch + 1,total_train_loss))
                profile.write(f"[{epoch}/{hyper_params['epochs']}] : [{i}/{n_batches}] : loss/ running_loss :{loss.item()}/{running_loss}]\n")
                print(f"[Train Epoch : {epoch}/{hyper_params['epochs']} [{i * hyper_params['bs']}/{len(dataloader_train.dataset)}  ({100*i/len(dataloader_train):.4f}%)]\tLoss: {loss.item():.6f} ]")
            gc.collect()

        epoch_loss = running_loss/n_batches  #Total loss for one epoch
        training_loss_list.append(epoch_loss)

        print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_loss))
        epochTimeEnd = time.time()-epochStartTime
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epochTimeEnd // 60, epochTimeEnd % 60))
        print('-' * 25)
        #Save the model
        torch.save(model.state_dict(),'../trained_model_trans_lrn_4_4_raw_n_clean/bt_ws_trns_lrn_without_change_phNN_28oct_lr_{}_epoch_{}_batch_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], hyper_params["bs"], epoch + 1,epoch_loss))
        #torch.save(transfr_learn.state_dict(),'../trained_models_transfer_learn/bt_ws_transfer_learn_phNN_lr_{}_epoch_{}_batch_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], hyper_params["bs"], epoch + 1,epoch_loss))
        #torch.save(remvd_last_layer_model.state_dict(),'../trained_models_transfer_learn/bt_ws_rmvd_last_layer_transfer_learn_phNN_lr_{}_epoch_{}_batch_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'],epoch+1, hyper_params["bs"],epoch_loss))
        model.eval()
        count = 0
        correct = 0
        n_batches_dev = len(dataloader_dev)

        print("range of dataloader in dev  ", n_batches_dev)
        for i, data in enumerate(dataloader_dev):
            #print("INSIDE enumerate")
            t1 = time.time()
            inp = torch.reshape(data[0], (-1, 360)).cuda()
            #inp = torch.reshape(data[0], (-1, 440)).cuda()
            lab = torch.reshape(data[1], (-1, 42)).cuda()
            if i == 0: print(inp.shape, inp.device, lab.shape, lab.device)

            inputs.requires_grad = False

            # Data feed-forward through the network
            outputs = model(inp)
            count += inp.shape[0]
            lab_max, lab_idx = torch.max(lab,1)
            max_val, max_idx = torch.max(outputs,1)
            correct += (max_idx  == lab_idx).float().sum()
            dev_loss = criterion(outputs, lab)
            running_dev_loss += dev_loss.item()

            # Accumulate loss per batch
            running_loss += loss.item()
            if i % 200 == 0:
                profile_dev.write(f"[{epoch}/{hyper_params['epochs']}] : [{i}/{n_batches_dev}] :loss/ running_loss {loss.item()}/{running_loss} ]\n")
                print(f"[Dev Epoch : {epoch}/{hyper_params['epochs']} [{i * hyper_params['bs']}/{len(dataloader_dev.dataset)}  ({100*i/len(dataloader_dev):.4f}%)]\tLoss: {loss.item():.6f} ]")
            gc.collect()
        print("Accuary = {:.6}".format(correct/count))
        epoch_dev_loss = running_dev_loss/n_batches_dev  #Total loss for one epoch
        print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_loss))
        print("####### Dev Loss ########")
        print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_dev_loss))
        epochTimeEnd = time.time()-epochStartTime
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epochTimeEnd // 60, epochTimeEnd % 60))
        print('-' * 25)
        dev_loss_list.append(epoch_dev_loss)
        fig1 = plt.figure(1)
        epoch_count = range(1, len(training_loss_list) + 1)
        plt.plot(epoch_count, training_loss_list, 'r--',label='Train loss')
        plt.plot(epoch_count, dev_loss_list, 'b-',label='Dev loss')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graphi
        fig1.savefig(f'../plot_4_4_raw_n_clean/Bt_ws_trnsfr_lrn_all_Phone_NN_28oct_btsz_{hyper_params["bs"]}_lrn_{hyper_params["learning_rate"]}_training_validation_loss.png')
    profile.close()
    profile_dev.close()
