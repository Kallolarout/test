import torch
from torch import nn
import torch.nn.functional as F

class Phone_NN(nn.Module):

    def __init__(self):
        super(Phone_NN, self).__init__()

        self.fc1 = nn.Linear(360, 256)
        self.sig1= nn.Sigmoid()

        self.fc2 = nn.Linear(256, 256)
        self.sig2= nn.Sigmoid()

        self.fc3 = nn.Linear(256, 256)
        self.sig3 = nn.Sigmoid()

        self.fc4 = nn.Linear(256, 256)
        self.sig4 = nn.Sigmoid()

        self.fc5 = nn.Linear(256, 42)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.sig1(out)

        out = self.fc2(out)
        out = self.sig2(out)

        out = self.fc3(out)
        out = self.sig3(out)

        out = self.fc4(out)
        out = self.sig4(out)

        out = self.fc5(out)
        out = self.softmax(out)

        return out

class Transfer_Learn_nochange_inphnnn(nn.Module):
    def __init__(self):
        super(Transfer_Learn_nochange_inphnnn,self).__init__
        self.model = Phone_NN()
        self.model.load_state_dict(torch.load("/home/paperspace/kws/trained_models/model_learning_rate_0.0001_epoch_100_epoch_loss_0.043774982606348746.pt"))
        for param in self.model.parameters():
            param.requires_grad = True

        def forward(self,x):
            out = self.model(x)
            return out



class Transfer_Learn1(nn.Module):
    def __init__(self):
        super(Transfer_Learn1, self).__init__()
        self.phone_NN = Phone_NN()
        self.Transfer_Learn.load_state_dict(torch.load("../src_5_sep_2019_bs/bt_ws_trns_lrn_without_change_phNN_lr_1e-05_epoch_16_batch_300_epoch_loss_0.013335964118915404.pt"))
        self.Transfer_Learn = orch.nn.Sequential(*list(self.phone_NN.children())[:-1])
        for param in self.Transfer_Learn.parameters():
            param.requires_grad = True

    def forward(self,x):
        out = self.Transfer_Learn(x)
        return out

class word_NN_only(nn.Module):
    def __init__(self):
        super(word_NN_only,self).__init__()
        self.fc1 = nn.Linear(5964,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        #self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256,64)
        self.bn3 = nn.BatchNorm1d(64)
        #self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64,2)
        #self.bn4 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.drop1(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)
        #x1 = self.drop2(x1)
        x1 = self.fc3(x1)
        x1 = self.relu(x1)
        x1 = self.bn3(x1)
        #x1 = self.drop3(x1)
        x1 = self.fc4(x1)
        x2 = self.softmax(x1)
        return x2


class word_NN_temp_combine_class(nn.Module):
    def __init__(self):
        super(word_NN_temp_combine_class,self).__init__()
        self.phone_NN = Phone_NN()
        self.phone_NN.load_state_dict(torch.load("/home/paperspace/kws/ok_huddle_kws/trained_model_trans_lrn_4_4_raw_n_clean/bt_ws_trns_lrn_without_change_phNN_28oct_lr_1e-05_epoch_16_batch_460_epoch_loss_0.021421159939159214.pt"))
        #self.phone_NN.load_state_dict(torch.load("/home/paperspace/kws/ok_huddle_kws/trained_model_trans_lrn_4_4/bt_ws_trns_lrn_without_change_phNN_22oct_lr_1e-05_epoch_8_batch_645_epoch_loss_0.020755924160281818.pt")) ## only raw waveforms transfer learned
        self.Phn_NN_transfer_learn = torch.nn.Sequential(*list(self.phone_NN.children())[:-1])
        #self.phone_NN.load_state_dict(torch.load("/home/paperspace/kws/ok_huddle_kws_global_mean/trained_model_trans_lrn_4_4/bt_ws_trns_lrn_without_change_phNN_22oct_lr_0.0001_epoch_64_batch_1000_epoch_loss_0.016122401829946925.pt"))
        for param in self.Phn_NN_transfer_learn.parameters():
            param.requires_grad = True
        #self.maxpool1d = nn.MaxPool1d(2, stride=1)
        self.word_NN_only = word_NN_only()


    def forward(self,x):
        #x1 = self.Phn_NN_transfer_learn(x)   #updating the initial layers also
        frm_list = []
        #print("inside the forward function of word nn : x.shape[1] ",x.shape[1])
        for frm_idx in range(x.shape[1]):
            inp = x[:,frm_idx,:]
            y = self.Phn_NN_transfer_learn(inp)
            frm_list.append(y)
        frm_list = torch.cat(frm_list,1)
        #frm_list = frm_list.view(frm_list.shape[0],1,frm_list.shape[1])
        #frm_list_max_pool = self.maxpool1d(frm_list)
        #frm_list_max_pool = frm_list_max_pool.view(frm_list_max_pool.shape[0],frm_list_max_pool.shape[2])
        #out = self.word_NN_only(frm_list_max_pool)
        out = self.word_NN_only(frm_list)
        return out,self.word_NN_only,self.Phn_NN_transfer_learn



class Word_NN(nn.Module):
    def __init__(self):
        super(Word_NN, self).__init__()
        self.model_phone_NN = Phone_NN()
        #self.model_phone_NN.load_state_dict(torch.load("../src_5_sep_2019_bs/bt_ws_trns_lrn_without_change_phNN_lr_1e-05_epoch_16_batch_300_epoch_loss_0.013335964118915404.pt"))
        self.model_phone_NN.load_state_dict(torch.load("../trained_model_trans_lrn/bt_ws_trns_lrn_without_change_phNN_22oct_lr_0.0001_epoch_64_batch_1000_epoch_loss_0.016122401829946925.pt"))
        #self.model_phone_NN.load_state_dict(torch.load("../trained_model_trans_lrn/bt_ws_trns_lrn_without_change_phNN_19sep_lr_1e-05_epoch_2_batch_121_epoch_loss_0.01850499970876359.pt"))
        self.model_phone_NN = torch.nn.Sequential(*list(self.model_phone_NN.children())[:-1])

        #print("############## AFTER REMOVING SOFTMAX ####",self.Transfer_Learn_nochange_inphnnn_rmvd)
        for param in self.model_phone_NN.parameters():
            param.requires_grad = True
        print("TTTTTTTTTTTTTTTTTTTTTTT ",self.model_phone_NN)
        self.fc1 = nn.Linear(5880,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,2)
        #self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        frm_list = []
        #print("inside the forward function of word nn : x.shape[1] ",x.shape[1])
        for frm_idx in range(x.shape[1]):
            inp = x[:,frm_idx,:]
            y = self.model_phone_NN(inp)
            frm_list.append(y)
        frm_list = torch.cat(frm_list,1)

        x1 = self.fc1(frm_list)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        #x2 = self.fc3(x1)
        x2 = self.softmax(x1)
        return x2, self.model_phone_NN ### upgating weight without changing any layer
