import torch
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import time
import gc

def train_word_nn(dataloader_train, dataloader_dev, model, hyper_params, device):
    model = model.cuda()
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters in network = {}'. format(total_num_params))
    print('Total number of trainable  parameters in network = {}'. format(total_num_trainable_params))
    #Loss criterion
    #criterion = torch.nn.BCELoss().to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    date = datetime.now()
    time_now =time.time()
    if hyper_params.get('optimizer') == 'Adam':
        #optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get('learning_rate'))
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get('learning_rate'), weight_decay = 1e-3)
        print("Adam optimizer set")
    else:
        print("Please choose proper optimizer for training")

    profile = open("../meta/profile_word_nn.time", "w")
    profile_dev = open("../meta/profile_dev_word_nn.time", "w")

    training_loss_list = []
    dev_loss_list = []
    for epoch in range(hyper_params['epochs']):
        epochStartTime = time.time()
        running_loss = 0.0
        running_loss_dev = 0.0
        #n_batches = len(train_loader)
        batch_size = hyper_params['bs']
        total_batches_train = len(dataloader_train)
        print("total_batches_train  ", total_batches_train)
        model.train()
        for i, batch in enumerate(dataloader_train):
            #print("INSIDE enumerate")
            t1 = time.time()
            inputs = batch[0].cuda()
            #labels = torch.reshape(batch[1], (-1, 2)).cuda()
            labels = batch[1].float().cuda()
            #print(labels)
            if i == 0: print(inputs.shape, inputs.device, labels.shape, labels.device)

            inputs = inputs.requires_grad_()
            # criterion.to(device)
            optimizer.zero_grad()

            outputs,model_word_NN, model_phone_NN = model(inputs)
            #labels = labels.view(-1).long()
            # Predicted class is the one with maximum probability
            outputs = outputs.view(outputs.shape[0],1,outputs.shape[1])
            # Finding the MSE
            loss = criterion(outputs, labels)
            loss.backward()

            # Update the network parameters
            optimizer.step()
            # Accumulate loss per batch
            running_loss += loss.item()
            profile.write(f"[{epoch}/{hyper_params['epochs']}] : [{i}/{total_batches_train}] : {loss.item()}/{running_loss} : Loss {loss.item():.6f}\n")
            if i %500 == 0: print(f"[Train Epoch: {epoch}/{hyper_params['epochs']}] : [{i*hyper_params['bs']}/{len(dataloader_train.dataset)} ({100*i/len(dataloader_train):.4f}%)]\tLoss: {loss.item():.6f} ]")
            gc.collect()
        epoch_loss = running_loss/total_batches_train #Total loss for one epoch
        #train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graphi

        print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_loss))
        epochTimeEnd = time.time()-epochStartTime
        training_loss_list.append(epoch_loss)
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epochTimeEnd // 60, epochTimeEnd % 60))
        print('-' * 55)
        #Save the model
        torch.save(model_word_NN.state_dict(),'../trained_model_word_nn_raw_n_clean_4_4/word_acoq_wdcy_NN_transf_22oct_tr_lr_{}_epoch_{}_batchsz_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], epoch + 1,hyper_params['bs'],epoch_loss))
        torch.save(model_phone_NN.state_dict(),'../trained_model_word_nn_raw_n_clean_4_4/Updated_transf_phnnn_acoq_wdcy_22oct_true_lr_{}_epoch_{}_batchsz_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], epoch + 1,hyper_params['bs'],epoch_loss))
        #torch.save(phnn_weight_fridge.state_dict(),'../trained_models_word_nn_utt_cmvn/Updated_transf_phnnn_rmvd_last2_model_lr_{}_epoch_{}_batchsz_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], epoch + 1,hyper_params['bs'],epoch_loss))
        #torch.save(phnn_weight_fridge.state_dict(),'../trained_models_word_nn_utt_cmvn/Updated_transf_phnnn_weight_fridge_model_lr_{}_epoch_{}_batchsz_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], epoch + 1,hyper_params['bs'],epoch_loss))
        #torch.save(transf_lrn.state_dict(),'../trained_models_word_nn_utt_cmvn/Updated_transf_phn_model_lr_{}_epoch_{}_batchsz_{}_epoch_loss_{}.pt'.format( hyper_params['learning_rate'], epoch + 1,hyper_params['bs'],epoch_loss))

        model.eval()
        num_dev = len(dataloader_dev)
        t2 = time.time()
        count = 0
        correct = 0
        not_correct = 0
        TP = 0
        FP = 0
        FN = 0
        for i, data in enumerate(dataloader_dev):
            inp = data[0].cuda()
            #lab = batch[1].cuda()
            lab = data[1].float().cuda()
            out,_,_= model(inp)
            out = out.view(out.shape[0],1,out.shape[1])
            #print(out.shape)
            loss_dev = criterion(out,lab)
            #print("labels shape",lab.shape)
            out_max,out_idx = torch.max(out,2)
            #print("out ", out)
            #print("lab ", lab.shape)
            #print("out_max ", out_max)
            #print("out_idx ", out_idx)
            lab_max,lab_idx = torch.max(lab,2)
            #print("lab_max ", lab_max)
            #print("lab_idx ", lab_idx.shape)
            #exit()
            count += inp.shape[0]
            correct += ( out_idx  == lab_idx  ).float().sum()
            not_correct += ( out_idx  != lab_idx  ).float().sum()
            for k in range(lab.shape[0]):
                if lab_idx[k] == 1 and out_idx[k] == 1 :
                    TP +=1
                elif lab_idx[k] == 1 and out_idx[k] == 0 :
                    FN += 1

                elif lab_idx[k] == 0 and out_idx[k] == 1 :
                    FP += 1


            running_loss_dev += loss_dev.item()
            profile_dev.write(f"[{epoch}/{hyper_params['epochs']}] : [{i}/{num_dev}] : {loss.item()}/{running_loss} \n")
            if i %500 == 0:  print(f"[Dev Epoch: {epoch}/{hyper_params['epochs']}] : [{i*hyper_params['bs']}/{len(dataloader_dev.dataset)} ({100*i/     len(dataloader_dev):.4f}%)]\tLoss: {loss_dev.item():.6f} ]")
        epoch_loss_dev = running_loss_dev/num_dev
        print("Accuary = {:.6}".format(correct/count))
        print("False (+ve as -ve & -ve as +ve) = {:.6}".format(not_correct/count))
        print("#"*55, "loss_dev")
        print("True positive ", TP)
        print("False positive ", FP)
        print("False negative ", FN)
        print("#"*55, "loss_dev")
        print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_loss_dev))
        epochTimeEnd = time.time()- t2
        dev_loss_list.append(epoch_loss_dev)
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epochTimeEnd // 60, epochTimeEnd % 60))
        fig1 = plt.figure(1)
        epoch_count = range(1, len(training_loss_list) + 1)
        plt.plot(epoch_count, training_loss_list, 'r--',label='Train loss')
        plt.plot(epoch_count, dev_loss_list, 'b-',label='Dev loss')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.show();
        fig1.savefig(f'../trained_model_word_nn_raw_n_clean_4_4/Word_NN_transf_acoq_27oct_wdcy_true_btsz_{hyper_params["bs"]}_lrn_{hyper_params["learning_rate"]}training_validation_loss.png')
    profile.close()
    profile_dev.close()
