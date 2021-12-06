import torch
import torch.utils.data
import multiprocessing
import pandas as pd
import numpy as np
import soundfile as sf
import textgrid
import decimal
import os
import time

class PHlabelDS(torch.utils.data.Dataset):
    def __init__(self, phlabel_paths):
        """
        """
        self.phlabel_paths = phlabel_paths # list of phlabels paths

    def __len__(self):
        return len(self.phlabel_paths)

    def __getitem__(self, idx):

        y = torch.load(self.phlabel_paths[idx])
        #y = y.cuda()
        f_name = self.phlabel_paths[idx].split("/")[-1].split(".")[0]

        return(y, f_name)

class wordNN(torch.utils.data.Dataset):
    def __init__(self, filename_csv):
        """
        word NN training script
        filename_csv : (str) conating both feature and label
        """
        self.filename_csv = filename_csv
        self.files = np.array(pd.read_csv(self.filename_csv, header = None)).squeeze()
        #print(self.files.shape[0])

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        input_path = self.files[idx][0]
        #print(input_path)
        label_path = self.files[idx][1]
        #print(label_path)
        m_input = torch.load(input_path)
        m_label = torch.load(label_path)
        m_label = m_label.view(1,m_label.shape[0])
        #print("m_label shape",m_label.shape)
        #print("m_feat shape",m_input.shape)

        #return input_path, label_path
        return m_input, m_label



class wordNN_dev(torch.utils.data.Dataset):
    def __init__(self, filename_csv):

        """
        word NN training script
        filename_csv : (str) conating both feature and label
        """
        self.filename_csv = filename_csv
        self.files = np.array(pd.read_csv(self.filename_csv, header = None)).squeeze()
        #print(self.files.shape[0])

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        input_path = self.files[idx][0]
        #print(input_path)
        label_path = self.files[idx][1]
        #print(label_path)
        m_input = torch.load(input_path)
        m_label = torch.load(label_path)
        m_label = m_label.view(1,m_label.shape[0])
        #print("m_label shape",m_label.shape)
        #print("m_feat shape",m_input.shape)

        return m_input, m_label


class Transfer_wordNN(torch.utils.data.Dataset):
    def __init__(self, filename_csv):
        """
        word NN training script
        filename_csv : (str) conating both feature and label
        """
        self.filename_csv = filename_csv
        self.files = np.array(pd.read_csv(self.filename_csv, header = None)).squeeze()
        #print(self.files.shape[0])

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        input_path = self.files[idx][0]
        #print(input_path)
        label_path = self.files[idx][1]
        #print(label_path)
        m_input = torch.load(input_path)
        m_label = torch.load(label_path)
        m_label = m_label.view(1,m_label.shape[0])
        #print("m_label shape",m_label.shape)
        #print("m_feat shape",m_input.shape)

        return m_input, m_label


class Transfer_wordNN_dev(torch.utils.data.Dataset):
    def __init__(self, filename_csv):
        """
        word NN training script
        filename_csv : (str) conating both feature and label
        """
        self.filename_csv = filename_csv
        self.files = np.array(pd.read_csv(self.filename_csv, header = None)).squeeze()
        #print(self.files.shape[0])

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        input_path = self.files[idx][0]
        #print(input_path)
        label_path = self.files[idx][1]
        #print(label_path)
        m_input = torch.load(input_path)
        m_label = torch.load(label_path)
        m_label = m_label.view(1,m_label.shape[0])
        #print("m_label shape",m_label.shape)
        #print("m_feat shape",m_input.shape)

        return m_input, m_label
class KWSTrainDS(torch.utils.data.Dataset):
    def __init__(self,filename_csv):
        """
        KWS Training script
        """
        self.filename_csv =  filename_csv
        self.files = np.array(pd.read_csv(self.filename_csv, header = None)).squeeze()


    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        input_path = self.files[idx][0]
        #print("input_path ",input_path[0])
        label_path = self.files[idx][1]
        m_input = torch.load(input_path)
        m_label = torch.load(label_path)
        return m_input , m_label

class Transfr_TrainBatchDS(torch.utils.data.DataLoader):
    def __init__(self, f_csv):
        self.f_csv = f_csv
        self.files = np.array(pd.read_csv(self.f_csv, header=None)).squeeze()
    def __len__(self):
        return self.files.shape[0]
    def __getitem__(self, idx):
        x_data = torch.load(self.files[idx])
        return x_data[0], x_data[-1]

class Transfr_DevBatchDS(torch.utils.data.DataLoader):
    def __init__(self, f_csv):
        self.f_csv = f_csv
        self.files = np.array(pd.read_csv(self.f_csv, header=None)).squeeze()
    def __len__(self):
        return self.files.shape[0]
    def __getitem__(self,idx):
        x_data = torch.load(self.files[idx])
        return x_data[0], x_data[-1]
class KWSTrainBatchDS(torch.utils.data.DataLoader):
    def __init__(self, f_csv):
        self.f_csv = f_csv
        self.files = np.array(pd.read_csv(self.f_csv, header=None)).squeeze()
    def __len__(self):
        return self.files.shape[0]
    def __getitem__(self, idx):
        x_data = torch.load(self.files[idx])
        return x_data[0], x_data[-1]

class KWSDevBatchDS(torch.utils.data.DataLoader):
    def __init__(self, f_csv):
        self.f_csv = f_csv
        self.files = np.array(pd.read_csv(self.f_csv, header=None)).squeeze()
    def __len__(self):
        return self.files.shape[0]
    def __getitem__(self,idx):
        x_data = torch.load(self.files[idx])
        return x_data[0], x_data[-1]



class KWSTrainDS_old(torch.utils.data.Dataset):
    def __init__(self, input_dir, label_dir, filename_csv, device, input_ext=".featTensor", label_ext=".probTensor"):
        """
        KWS Training dataset class
        """
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.filename_csv = filename_csv
        self.input_ext = input_ext
        self.label_ext = label_ext
        self.fnames = pd.read_csv(self.filename_csv, sep=",", header=None).values
        self.device = device

    def __len__(self):

        return self.fnames.shape[1]

    def __getitem__(self, idx):
        input_pth = os.path.join(self.input_dir, self.fnames[0][idx]+self.input_ext)
        label_pth = os.path.join(self.label_dir, self.fnames[0][idx]+self.label_ext)
        m_input = torch.load(input_pth)
        #m_input = torch.Tensor(m_input, dtype=torch.FloatTensor, device="cuda:0")
        m_label = torch.load(label_pth)
        #print(m_input)
        sample = {"input": m_input, "label": m_label,}
        return sample

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, train_phlabel_paths):
        """
        wav_paths: list of strings (wav file paths)
        textgrid_paths: list of strings (textgrid for each wav file)
        Sy_phoneme: list of strings (all possible phonemes)
        Sy_word: list of strings (all possible words)
        config: Config object (contains info about model and training)
        """
        self.wav_paths = wav_paths # list of wav file paths
        self.train_phlabel_paths = train_phlabel_paths # list of textgrid file paths
        self.length_mean = 2.25
        self.length_var = 1
        #self.Sy_word = Sy_word
        self.phone_downsample_factor = 1
        self.word_downsample_factor = 1
        self.fs = 16000
        # self.loader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsASR())

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):

        x, _ = sf.read(self.wav_paths[idx])
        x = torch.tensor(x)
        y = torch.load(self.train_phlabel_paths[idx])
        f_name = self.wav_paths[idx].split("/")[-1].split(".")[0]

        #print("Data size in samples ", x.shape)

        #print(x.shape, y.shape)
        #print(x)
        #print("#"*100)
        return(x, y, f_name)


class CollateWavsASR:
    def __call__(self, batch):
        """
        batch: list of tuples (input wav, phoneme labels, word labels)
        Returns a minibatch of wavs and labels as Tensors.
        """
        x = []; y_phoneme = []; y_word = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_,y_phoneme_, y_word_ = batch[index]

            x.append(torch.tensor(x_).float())
            y_phoneme.append(torch.tensor(y_phoneme_).long())
            y_word.append(torch.tensor(y_word_).long())

        # pad all sequences to have same length
        T = max([len(x_) for x_ in x])
        U_phoneme = max([len(y_phoneme_) for y_phoneme_ in y_phoneme])
        U_word = max([len(y_word_) for y_word_ in y_word])
        for index in range(batch_size):
            x_pad_length = (T - len(x[index]))
            x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

            y_pad_length = (U_phoneme - len(y_phoneme[index]))
            y_phoneme[index] = torch.nn.functional.pad(y_phoneme[index], (0,y_pad_length), value=-1)

            y_pad_length = (U_word - len(y_word[index]))
            y_word[index] = torch.nn.functional.pad(y_word[index], (0,y_pad_length), value=-1)

        x = torch.stack(x)
        y_phoneme = torch.stack(y_phoneme)
        y_word = torch.stack(y_word)

        return (x,y_phoneme, y_word)
