import sys,os
import glob
import torch

from transfer_phonenn_360hr_kaldi import Phone_NN
#from transfer_phonenn import Transfer_Learn_nochange_inphnnn
#from transfer_phonenn import transf_combine_class
import helper as hp
import data as m_data
#from train_model_transfer_learn import train_transfer_learn    ## for frame wise training
from train_model_transfer_learn_batchwise import train_transfer_learn_bs     ## for batchwise learning
from phoneme_mapping_transfer_learn import phoneme_map

import numpy as np
import feature_extraction as fe
import pandas as pd


def main():

    train_params = {'learning_rate': 0.00001,
                    'optimizer': 'Adam',
                    'epochs': 1000,
                    'bs': 32,                ##### for frame wise high value else 32 or 64 for batch wise
                    'n_workers': 4,
                    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')


    uttr_fnames = np.array(pd.read_csv("/home/paperspace/kws/ok_huddle_kws/data/splice_feats_clean_4_4.scp", header=None, delimiter=" ")[0])
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
    #   uttr_fnames[i] = f"/home/paperspace/kws/ok_huddle_kws/data/TextGrid_file/"\
    #                 + f"{utt_name}/"\
    #                 + f"{last_utt.split('.')[0]}.TextGrid"
    #                 #+ f"{n.rsplit('-')[1]}.TextGrid"
    #                 #+ f"{n.split('-')[1]}/"\
    #   print(" ################ uttr_fnames[i]  ############## ",uttr_fnames[i])
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

    #phoneme_map(uttr_fnames, Sy_phoneme, f"/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_amp_phn_4_4_phlabel/")

    #exit()
    ###############################################################
    ######### probability extraction ###############################
    ################################################################
    #print("uttr fnames", uttr_fnames)
    #train_phlabel_paths = [f"/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_amp_phn_4_4_phlabel/{p.split('/')[-1].split('.')[0]}.phlabelTensor" for p in uttr_fnames]

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
    ##device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Running on the {device}')
    ######################################################################################
    ############ Training without batchwise combining as earlier for libri speech  #######
    #####################################################################################
    train_ds = m_data.Transfr_TrainBatchDS("/home/paperspace/kws/ok_huddle_kws/data/batch_total_raw_clean_phn_feat_prob_all_4_4_80.csv")
    #train_ds = m_data.Transfr_TrainBatchDS("/home/paperspace/kws/ok_huddle_kws/data/train_raw_feat_prob_4_4_all_80.csv")
    ##train_ds = m_data.wordNN("/home/paperspace/kws/data/speech_cmd_alexa_nonalexa_libri.csv")
    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    dev_ds = m_data.Transfr_DevBatchDS("/home/paperspace/kws/ok_huddle_kws/data/batch_total_raw_clean_phn_feat_prob_all_4_4_20.csv")
    #dev_ds = m_data.Transfr_DevBatchDS("/home/paperspace/kws/ok_huddle_kws/data/train_raw_feat_prob_4_4_all_20.csv")
    dev_dl = torch.utils.data.DataLoader(dataset=dev_ds, batch_size=train_params["bs"], shuffle=True, num_workers=train_params["n_workers"])
    phone_Net = Phone_NN().to(device)
    phone_Net.load_state_dict(torch.load("/home/paperspace/kws/ok_huddle_kws/kaldi_model_360.pt"))
    for param in phone_Net.parameters():
        param.requires_grad = True
    ##print(transfer_learn)
    print(phone_Net)
    #train_transfer_learn(dataloader_train=train_dl, dataloader_dev=dev_dl, model=phone_Net, hyper_params=train_params, device=device)

    train_transfer_learn_bs(dataloader_train=train_dl, dataloader_dev=dev_dl, model=phone_Net, hyper_params=train_params, device=device)
    print("#"*100)
    exit()

    ######################################################################################
    ################### Training batch similar to libri speech ############################
    ######################################################################################

    #train_ds = m_data.Transfr_TrainBatchDS("/home/paperspace/kws/data/alexa_xmos_rec_ampl_minus_10_15/batch_alexa_data_xmos_rec_ampl_minus10_15_feat_prob_all_80.csv")
    #train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_params["bs"], shuffle=False, num_workers=train_params["n_workers"])
    #dev_ds = m_data.Transfr_DevBatchDS("/home/paperspace/kws/data/alexa_xmos_rec_ampl_minus_10_15/batch_alexa_data_xmos_rec_ampl_minus10_15_feat_prob_all_20.csv")
    #dev_dl = torch.utils.data.DataLoader(dataset=dev_ds, batch_size=train_params["bs"], shuffle=False, num_workers=train_params["n_workers"])

    ##################
    #####  traing with last 2 update layer ####
    ##################

    #transfer_learn_bs = transf_combine_class().to(device)
    #train_transfer_learn_bs(dataloader_train=train_dl, dataloader_dev=dev_dl, model=transfer_learn_bs, hyper_params=train_params, device=device)
    #print(transfer_learn_bs)

    phone_Net = Phone_NN().to(device)
    for param in phone_Net.parameters():
        param.requires_grad = True


    #transfer_learn_without_chnage = Transfer_Learn_nochange_inphnnn().to(device)
    #train_transfer_learn_bs(dataloader_train=train_dl, dataloader_dev=dev_dl, model=phone_Net, hyper_params=train_params, device=device)
    train_transfer_learn(dataloader_train=train_dl, dataloader_dev=dev_dl, model=phone_Net, hyper_params=train_params, device=device)





    print("#"*100)




if __name__ == '__main__':
    main()


