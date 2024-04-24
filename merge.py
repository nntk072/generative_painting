# Read through the directory psnr and ssim, merge all the files into each directory into 1 numpy array and save them as psnr.npy and ssim.npy respectively.
import os
import numpy as np

def merge_psnr(directory):
    psnr_list = []
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            psnr_list.append(np.load(os.path.join(directory, file)))
    np.save("psnr.npy", np.concatenate(psnr_list))
    
def merge_ssim(directory):
    ssim_list = []
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            ssim_list.append(np.load(os.path.join(directory, file)))
    np.save("ssim.npy", np.concatenate(ssim_list))
    
if __name__ == "__main__":
    merge_psnr("psnr")
    merge_ssim("ssim")