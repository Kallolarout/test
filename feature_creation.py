import kaldi_io
import os,sys
import torch

#input argument is scp file

inp1 = sys.argv[1]
for key,mat in kaldi_io.read_mat_scp(inp1):
    for i in range(mat.shape[0]):
        print(key , "  ")
        feature = torch.from_numpy(mat[i])
        torch.save(feature, f"/home/paperspace/kws/data/alexa_data_test/feature/{key}_{i}.featTensor")
