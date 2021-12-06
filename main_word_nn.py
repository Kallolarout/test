import sys,os
import glob
import torch
#from transfer_phonenn import Word_NN,Transfer_Learn_nochange_inphnnn
#from transfer_phonenn import word_NN_temp_combine_class
from transfer_phonenn_360hr_kaldi import word_NN_temp_combine_class
#from phonenn import Phone_NN, Word_NN
import helper as hp
import data as m_data
from train_model_word_nn import train_word_nn

import numpy as np
import pandas as pd


def main():

    train_params = {'learning_rate': 0.00001,
                    'optimizer': 'Adam',
                    'epochs': 500,
                    'bs': 32,
                    'n_workers': 16,
                    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on the {device}')
    train_ds = m_data.wordNN("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_SET2_HCU_libri_neg_all_4_4_80.csv")
    #train_ds = m_data.wordNN("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_libri_neg_all_4_4_80.csv")
    #train_ds = m_data.wordNN("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_data_ok_huddle_pro_clean_SET2_HCU_cmvn_word_nn_feat_prob_4_4_all_80.csv")
    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    dev_ds = m_data.wordNN_dev("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_SET2_HCU_libri_neg_all_4_4_20.csv")
    #dev_ds = m_data.wordNN_dev("/home/paperspace/kws/ok_huddle_kws_global_mean/data/word_nn/train_word_nn_pos_neg_libri_4_4_all_20.csv")
    dev_dl = torch.utils.data.DataLoader(dataset=dev_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    ################################################
    ################## Word NN #####################
    ################################################
    ##phone_Net = Phone_NN().to(device)
    #transfer_learnno_change = Transfer_Learn_nochange_inphnnn()
    #print("before ",transfer_learnno_change)
    #transfer_learnno_change.load_state_dict(torch.load("../trained_models_transfer_learn/bt_ws_trns_lrn_without_change_phNN_lr_1e-06_epoch_32_batch_300_epoch_loss_0.009451012973204937.pt"))
    #transfer_learnno_change =  torch.nn.Sequential(*list(transfer_learnno_change.model.children())[:-1])
    #print(" AFTER ",transfer_learnno_change)
    #word_Net = Word_NN(transfer_learnno_change)
    ##print(word_Net)
    word_Net = word_NN_temp_combine_class()
    train_word_nn(dataloader_train=train_dl, dataloader_dev=dev_dl, model=word_Net, hyper_params=train_params, device=device)

    print("#"*100)



if __name__ == '__main__':
    main()


