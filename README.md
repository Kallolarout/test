TDNN Paper implementaion 

A. Dataset creation 
 1. Depending on Keyword ("ALEXA", "OKAY HUDDL") length, choose the duration of keyword, according;y split the utterance

 2. Using Kaldi create the .scp file and generate feature using mfcc, fbank , splice with left & right features . Using feature_creation.py file convert it into tensor.
 
 3. helper.py is used for creating batch file , csv file and split it. 

  in main.py training script : 
  STEP1 : Map the TEXTGRID file path, create the phenome file from the TEXTGRID file using "phoneme_mapping.py" file.
  STEP2 : Probability extraction : which belongs to which phoneme.
          Divide the the phoneme file into subsequent frames of 25msec windows size and 10ms shift
  
  STEP3 : Use helper.py to create csv file. Then use batch_tensor_data_creation to make them in batch.
          Then use helper.py to split the csv into 80 and 20% for train and evaluation
   
  STEP4 : Train the phone model : input file will all librispeech and general phonenn training
  
  STEP6 : train_model_word_nn.py is used to train word nn. with input as keyword dataset
  
  
  TESTING : For offline testing "offline_test.py". Before proceeding to test create required file and call the correct funciton
  