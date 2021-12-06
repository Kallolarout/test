import os
import numpy as np
import pandas as pd
import gc
import torch

def temp_create_csv_file(inp_dir,label_dir):
    print("inside the create csv file")
    #inp_label_path = 'Database/Exp/phoneme_label_prob_100/'
    #inp_feat_path = 'Database/Exp/feature_100/'
    ln='\n'
    with open('Database/Exp/test_temp.csv','w') as csv_file:
        for i in os.listdir(label_dir):
            print(i)
            #file_path = os.path.isfile(inp_dir+i.split('.')[0]+".featTensor")
            #file_path = os.path.isfile(inp_dir+i.split('.')[0]+".pt")
            #print(f"file path :{file_path}")
            #file_path = os.path.isfile(f"{inp_dir}{i.split('.')[0]}.featTensor")
            #if file_path:
            #csv_file.write(f"{inp_dir}{i.split('.')[0]}.featTensor,{label_dir}{i}\n")
            csv_file.write(f"{inp_dir}{i.replace('label_1_42','feat_128_440')},{label_dir}{i}{ln}")
            #else:
            #   print("not der")
    csv_file.close()

def reorganization(root_input_path: str, root_label_path: str, spkr_file: str):
    print("inside the or")
    spkr_list = []
    spkr_file = pd.read_csv(spkr_file, header=None)
    for i in range(spkr_file.shape[0]):
        spkr = spkr_file[0][i]
        print("spkr name",spkr)
        #$FIND -name $FILE -type f -mtime $MDATE -exec mv -t  {} \;
        #tmp = "find -maxdepth 1 -name 'Database/Exp/feature_100_part1/'+str(spkr)+'*.jpg' -exec mv -t ../../dst/ {} +"
        #os.system(tmp)
        #os.system("find -wholename 'Database/Exp/feature_100_part1/"+str(spkr)+"*.featTensor'"+" -exec mv -t "+root_input_path+str(spkr)+" {} +")
        os.system("find -maxdepth 1 -name  "+"Database/Exp/feature_100_part1/+*.featTensor"+" -exec mv -t "+root_input_path+str(spkr)+" {} +")
        print("DONE", i)


    """
    csv_file = pd.read_csv(inp_csv_file,header=None)

    for i in range(csv_file.shape[0]):
        spkr = csv_file[0][i].split("/")[-1].split("-")[0]
        print(f"{spkr}")
        if spkr not in spkr_list:
            spkr_list.append(spkr)
            os.system(f"mkdir -p {root_input_path}{spkr}")
            os.system(f"mkdir -p {root_label_path}{spkr}")
            os.system(f"mv {csv_file[0][i]} {root_input_path}{spkr}")
            os.system(f"mv {csv_file[1][i]} {root_label_path}{spkr}")
            os.system(f"ls {root_input_path}{spkr} | wc -l")
            gc.collect()
        else:
            os.system(f"mv {csv_file[0][i]} {root_input_path}{spkr}")
            os.system(f"mv {csv_file[1][i]} {root_label_path}{spkr}")
            os.system(f"ls {root_input_path}{spkr} | wc -l")
            gc.collect()

    """

    print("**"*10)
def split_csv_file_train_dev(csv_file):
    xx = open(csv_file,'r')
    no_file_in70 = 0
    no_file_in30 = 0
    #csv_70 = open("/home/paperspace/kws/ok_huddle_kws/data/batch_total_raw_clean_phn_feat_prob_all_4_4_80.csv","w")
    #csv_30 = open("/home/paperspace/kws/ok_huddle_kws/data/batch_total_raw_clean_phn_feat_prob_all_4_4_20.csv","w")
    csv_70 = open("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_SET2_HCU_libri_neg_all_4_4_80.csv","w")
    csv_30 = open("/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_SET2_HCU_libri_neg_all_4_4_20.csv","w")
    #csv_70 = open("/home/paperspace/kws/data/alexa_nonalexa_libri_data_all_80.csv","w")
    #csv_30 = open("/home/paperspace/kws/data/alexa_nonalexa_libri_data_all_20.csv","w")
    #csv_70 = open("/home/paperspace/kws/data/alexa_non_alexa_libri_xmos_rec_speech_cmd_80.csv","w")
    #csv_30 = open("/home/paperspace/kws/data/alexa_non_alexa_libri_xmos_rec_speech_cmd_20.csv","w")
    for i, data in enumerate(xx):
        count = i%10
        if count <= 7:
            csv_70.write(data)
            no_file_in70 +=1
        else:
            csv_30.write(data)
            no_file_in30 +=1
    csv_70.close()
    csv_30.close()
    print("number of file in train : ", no_file_in70)
    print("number of file in dev : ", no_file_in30)

def making_128_feat_concat(inp_dir,lab_dir,inp_csv_file):
    csv_file = pd.read_csv(inp_csv_file, header=None)
    import time

    t_start = time.time()
    for i in range(csv_file.shape[0]):
        frm_cntr = i%256
        if frm_cntr == 0:
            first_x = torch.load(csv_file[0][i])
            first_x = torch.reshape(first_x,(1,440))

            first_y = torch.load(csv_file[1][i])
            first_y = torch.reshape(first_y,(1,42))
        elif frm_cntr == 1:
            second_x = torch.load(csv_file[0][i])
            second_x = torch.reshape(second_x,(1,440))
            stacked_feat = torch.cat((first_x,second_x),0)

            second_y = torch.load(csv_file[1][i])
            second_y = torch.reshape(second_y,(1,42))
            stacked_feat_y = torch.cat((first_y,second_y),0)
        else:
            current_feat = torch.load(csv_file[0][i])
            current_feat = torch.reshape(current_feat,(1,440))
            stacked_feat = torch.cat((stacked_feat,current_feat),0)

            current_feat_y = torch.load(csv_file[1][i])
            current_feat_y = torch.reshape(current_feat_y,(1,42))
            stacked_feat_y = torch.cat((stacked_feat_y,current_feat_y),0)


        #print(csv_file[0][i],"  ",csv_file[1][i])
        if frm_cntr == 0:
            if i == 0:
                continue
            else:
                print("Numer of frames are done",i)
                torch.save(stacked_feat,f"{inp_dir}{i}.featTensor")
                torch.save(stacked_feat_y,f"{lab_dir}{i}.featTensor")

                del stacked_feat, stacked_feat_y
                print(time.time()- t_start)
                t_start = time.time()




def creating_list(inp_csv_file):
    file_name = inp_csv_file.split('.csv')[0].split("/")[-1]
    tmp = file_name+"_list"
    print(tmp)
    tmp = []

    phoneme_label = open(inp_csv_file,'r')

def create_csv_file(out_csv_file,inp_path,lab_path):
    csv_file_out = open(out_csv_file,'w')

    for i in os.listdir(inp_path):
        print(i)
        csv_file_out.write(inp_path+"/"+i.strip()+","+lab_path+"/"+i.strip().replace(".featTensor",".probTensor")+"\n")
    csv_file_out.close()

def create_csv_file_temp(out_csv_file,inp_path,lab_path):
    csv_file_out = open(out_csv_file,'w')

    for i in os.listdir(inp_path):
        print(i)
        csv_file_out.write(inp_path+"/"+i.strip()+","+lab_path+"/"+i.strip().replace(".wav","").replace(".featTensor",".probTensor")+"\n")
    csv_file_out.close()
def create_csv_for_batch(inp_path,csv_file):
    csv_file_out=open(csv_file,'w')
    for i in os.listdir(inp_path):
        csv_file_out.write(i.strip()+"\n")
    csv_file_out.close()

def batch_tensor_data_creation(csv_file: str, bs: int, out_csv: str):
    import data as m_data
    import torch
    import time
    ds = m_data.KWSTrainDS(csv_file)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs)
    with open(out_csv, "w") as f:
        t0 = 0
        for i, batch in enumerate(dl):
            #print(type(batch), len(batch[0]))
            torch.save(batch, f"/home/paperspace/kws/ok_huddle_kws/data/batch_train_clean_data_4_4/{i}.batch")
            f.write(f"/home/paperspace/kws/ok_huddle_kws/data/batch_train_clean_data_4_4/{i}.batch\n")
            #print(f"delT_batch_dl : {time.time()-t0}")
            t0 = time.time()

        print(f"delT_batch_dl : {time.time()-t0}")

