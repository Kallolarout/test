import os,sys,glob
import torch
import numpy as np
import pandas as pd
from transfer_phonenn_360hr_kaldi import Phone_NN, Word_NN
import helper as hp
import kaldi_io

def main(x):
    #inp_feat_extraction(x) # it will return test.csv
    create_csv = csv_creat_fun(x)
    m = feed_forward()

def csv_creat_fun(x):
    csv_file =open("/home/paperspace/kws/temp_test/test.csv","w")
    for i in os.listdir(x):
        if i.endswith(".featTensor"):
            csv_file.write("/home/paperspace/kws/temp_test/"+i)
    csv_file.close()


def inp_feat_extraction(x):
    wav = open(x+"/wav.scp",'w')
    for i in os.listdir(x):
        if i.endswith('.wav'):
            wav.write(i+" "+x+"/"+i)
    wav.close()
    os.system("bash feature_extraction.sh " +x)
    os.system("python feat_flatten.py " + "/home/paperspace/kws/temp_test/feats.scp 5 5")
    #os.system("python feat_flatten.py " + "/home/paperspace/kws/temp_test/feats_cmvn.scp 5 5")

def feed_forward():
    word_Net = Word_NN()
    word_Net.eval()
    #word_Net.load_state_dict(torch.load("/home/paperspace/kws/trained_models_word_nn/model_learning_rate_1e-05_epoch_25_epoch_loss_0.584066495249167.pt"))
    word_Net.load_state_dict(torch.load("/home/paperspace/kws/trained_models_word_nn/model_learning_rate_1e-05_epoch_21_epoch_loss_0.5040671101638249.pt"))
    inp1 = torch.load(pd.read_csv("/home/paperspace/kws/temp_test/test.csv",header=None).squeeze())
    inp1 = inp1.view(1,70,440)
    out = word_Net(inp1)
    print(out[0])


if __name__ == '__main__':
    x= sys.argv[1]
    main(x)
