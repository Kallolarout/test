import sys,os
import glob
import torch

from transfer_phonenn import Phone_NN
import helper as hp
import data as m_data
from train_batchwise import train_bs     ## for batchwise learning
from phoneme_mapping_transfer_learn import phoneme_map

import numpy as np
import feature_extraction as fe
import pandas as pd


def main():

    train_params = {'learning_rate': 0.00001,
                    'optimizer': 'Adam',
                    'epochs': 1000,
                    'bs': 16,                ##### for frame wise high value else 32 or 64 for batch wise
                    'n_workers': 4,
                    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    ####################################################
    ##################### STEP1 ########################
    ####################################################
    
    
    #uttr_fnames = np.array(pd.read_csv("/home/paperspace/kws/ok_huddle_kws_global_mean/data/splice_feats_4_4.scp", header=None, delimiter=" ")[0])
    #print("########### utterance ###############  ",uttr_fnames[0])
    #for i, n in enumerate(uttr_fnames):
    #   x_split = n.split('-')
    #   utt_name = n.split('-')[0]
    #   last_utt = n.split('-')[-1]
    #   x_split_len = len(x_split)
    #   #print("x_split  ",x_split)
    #   #print(f"{last_utt.split('.wav')[0]}.TextGrid")
    #   #exit()
    #   #print("split_list  ",x_split)
    #   for k in range(x_split_len):
    #       if k >= 2:
    #           utt_name =utt_name +"-"+ str(x_split[k-1])
    #           #print("utterance name  ",utt_name)
    #       else:
    #           pass
    #   if utt_name.endswith('.wav'):
    #       temp_list = utt_name.split('-')
    #       # print("temp_list  ",temp_list)
    #       utt_name=""
    #       for ind in range(len(temp_list)-1):
    #           if ind == 0:
    #               utt_name=str(temp_list[ind])
    #           else:
    #               utt_name=utt_name+"-"+str(temp_list[ind])
    #               print("utterance name  ",utt_name)
    #               print("last utt  name  ",last_utt)

    #   #print("utt_name ",  utt_name)
    #   #x = n.split('-')[0]
    #   #uttr_fnames[i] = f"/home/paperspace/kws/ok_huddle_kws_global_mean/data/iiit_data_1515ms_TextGrid/"\
    #   uttr_fnames[i] = f"/home/paperspace/kws/ok_huddle_kws_global_mean/data/TextGrid_file/"\
    #                 + f"{utt_name}/"\
    #                 + f"{last_utt.split('.')[0]}.TextGrid"
    #                 #+ f"{n.rsplit('-')[1]}.TextGrid"
    #                 #+ f"{n.split('-')[1]}/"\
    #   #print(" ################ uttr_fnames[i]  ############## ",uttr_fnames[i])
    #   #exit()
    #if os.path.isfile("/home/paperspace/kws/assets/phonemes.txt"):
    #    Sy_phoneme = []
    #    with open("/home/paperspace/kws/assets/phonemes.txt", "r") as f:
    #        for line in f.readlines():
    #            if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
    #    num_phonemes = len(Sy_phoneme)
    #else:
    #    print("Getting vocabulary...")
    #    phoneme_counter = Counter()
    #    for path in valid_textgrid_paths:
    #        tg = textgrid.TextGrid()
    #        tg.read(path)
    #        phoneme_counter.update([phone.mark.rstrip("0123456789") for phone in tg.getList("phones")[0] if phone.mark != ''])
    #    Sy_phoneme = list(phoneme_counter)
    #    config.num_phonemes = len(Sy_phoneme)
    #    with open("/home/paperspace/kws/assets/phonemes.txt", "w") as f:
    #        for phoneme in Sy_phoneme:
    #            f.write(phoneme + "\n")
    #print("Sy_phoneme ",Sy_phoneme)
    #print("Sy_phoneme extraction Done.")

    #phoneme_map(uttr_fnames, Sy_phoneme, f"/home/paperspace/kws/ok_huddle_kws_global_mean/data/phlabel_train_data_ok_huddle/")

    #exit()
    
    ####################################################
    ##################### STEP2 ########################
    ####################################################
    
    ##print("uttr fnames", uttr_fnames)
    #train_phlabel_paths = [f"/home/paperspace/kws/ok_huddle_kws_global_mean/data/phlabel_train_data_ok_huddle_4_4/{p.split('/')[-1].split('.')[0]}.phlabelTensor" for p in uttr_fnames]

    #print(train_phlabel_paths[1])
    #print(train_phlabel_paths[10])
    #def phlabel_collate_fn(batch):
    #   labels = [item[0] for item in batch]
    #   f_names = [item[1] for item in batch]
    #   return [labels, f_names]

    #phlabel_extrxn_ds = m_data.PHlabelDS(train_phlabel_paths)
    #phlabel_extrxn_dl = torch.utils.data.DataLoader(dataset=phlabel_extrxn_ds, collate_fn=phlabel_collate_fn, batch_size=train_params["bs"], num_workers=train_params["n_workers"],drop_last=False)
    ##print(".-.-"*15)

    #fe.phlabel_prob_extract(dl=phlabel_extrxn_dl, device=device)
    #print("phlabel_prob_extract() DONE")
    #exit()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on the {device}')
    
    ###########################################################################################
    ########### STEP4 Training libri speech  #######
    ###########################################################################################
    #train_ds = m_data.Transfer_wordNN("/home/paperspace/kws/ok_huddle_kws_global_mean/data/batch_train_data_phonnn_feat_prob_all_80.csv")
    train_ds = m_data.Transfr_TrainBatchDS("/home/paperspace/kws/ok_huddle_kws_global_mean/data/batch_train_data_phonnn_feat_prob_all.csv")
    ##train_ds = m_data.wordNN("/home/paperspace/kws/data/speech_cmd_alexa_nonalexa_libri.csv")
    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    ##dev_ds = m_data.wordNN_dev("/home/paperspace/kws/data/alexa_nonalexa_libri_data_all_20.csv")
    #dev_ds = m_data.Transfer_wordNN_dev("/home/paperspace/kws/ok_huddle_kws_global_mean/data/batch_train_data_phonnn_feat_prob_all_20.csv")
    dev_ds = m_data.Transfr_DevBatchDS("/home/paperspace/kws/ok_huddle_kws_global_mean/data/batch_train_data_phonnn_feat_prob_all_20.csv")
    dev_dl = torch.utils.data.DataLoader(dataset=dev_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    phone_Net = Phone_NN().to(device)

    print(phone_Net)

    train_bs(dataloader_train=train_dl, dataloader_dev=dev_dl, model=phone_Net, hyper_params=train_params, device=device)
    print("#"*100)
    exit()

    print("#"*100)




if __name__ == '__main__':
    main()


