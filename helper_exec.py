import helper as hp
import os
import time

##################################################
###### Create csv file for both feat and prob ####
##################################################
#inp_path= "/home/paperspace/kws/data_utt/alexa_alignment/train_data_noise_feat/"
#lab_path="/home/paperspace/kws/data_utt/alexa_alignment/phlabel_train_data_noise_prob/"
#out_csv_file ="/home/paperspace/kws/ok_huddl_kws/data/ok_huddle_phonnn_feat_prob_all.csv"
#inp_path= "/home/paperspace/kws/ok_huddl_kws/data/ok_huddle_phonnn_feat/"
#lab_path="/home/paperspace/kws/ok_huddl_kws/data/ok_huddle_phonnn_prob/"
#hp.create_csv_file(out_csv_file,inp_path,lab_path)
#exit()

##################################################
########## batch tensor dat creation #############
##################################################


#inp_csv_file ="/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_cmvn_phn_nn_feat_prob_4_4_all.csv"
#out_csv_file = "/home/paperspace/kws/ok_huddle_kws/data/batch_train_clean_data_cmvn_phn_nn_feat_prob_4_4_all.csv"
#hp.batch_tensor_data_creation(inp_csv_file,256,out_csv_file)
#exit()

##########################
### SPLITING csv file ####
##########################

#inp_csv_file = "/home/paperspace/kws/ok_huddle_kws/data/batch_total_raw_clean_phn_feat_prob_all_4_4.csv"
#inp_csv_file = "/home/paperspace/kws/data/alexa_non_alexa_libri_xmos_rec_speech_cmd.csv"
inp_csv_file = "/home/paperspace/kws/ok_huddle_kws/data/word_nn/word_nn_ok_hudl_raw_pos_aco_neg_clean_pos_SET2_HCU_libri_neg_all_4_4.csv"
hp.split_csv_file_train_dev(inp_csv_file)
exit()

###############################
#### create csv file ##########
###############################
in_t = time.time()
out_csv_file ="/home/paperspace/kws/ok_huddle_kws/data/word_nn/data_ok_huddle_word_nn_SET2_HCU_feat_4_4_feat_prob_all.csv"
inp_path= "/home/paperspace/kws/ok_huddle_kws/data/word_nn/data_ok_huddle_word_nn_SET2_HCU_feat_4_4/"
lab_path="/home/paperspace/kws/ok_huddle_kws/data/word_nn/data_ok_huddle_word_nn_SET2_HCU_label_4_4/"
#out_csv_file ="/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_cmvn_phn_nn_feat_prob_4_4_all.csv"
#inp_path= "/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_amp_phn_feat_4_4/"
#lab_path="/home/paperspace/kws/ok_huddle_kws/data/train_clean_data_amp_phn_prob_4_4/"
#inp_path = "/home/paperspace/kws/data/alexa_xmos_rec_ampl_minus_10_15/alexa_data_xmos_rec_ampl_minus10_15_feat_word/"
#lab_path = "/home/paperspace/kws/data/alexa_xmos_rec_ampl_minus_10_15/alexa_data_xmos_rec_ampl_minus10_15_label_word/"
#inp_csv_file = "/home/paperspace/kws/data/libri_100hr/feat_2_10000.scp"
#out_csv_file = "/home/paperspace/kws/data/alexa_xmos_rec_ampl_minus_10_15/alexa_data_xmos_rec_ampl_minus10_15_feat_label_word_all.csv"

hp.create_csv_file(out_csv_file,inp_path,lab_path)
#hp.create_csv_file_temp(out_csv_file,inp_path,lab_path) ## only for xmos rec transfer learn feat and prob
#folder_name = "feature_100_part2"

#hp.create_csv_file_spkr(csv_file_name,inp_path,folder_name)

#hp.making_128_feat_concat(inp_path,lab_dir,csv_file_name)

print("time consumed :",time.time() - in_t)
exit()
'''
inp_path = "/home/paperspace/kws/data/libri_100hr/batch256_1_5000/"
csv_file_name ="batch256_1_5000.csv"
hp.create_csv_for_batch(inp_path,csv_file_name)'''
