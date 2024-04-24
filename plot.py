# Load the data from mask_percentage_list.npy as x axis, and psnr.npy and ssim.npy as y axis, plot the graph and save it in 2 images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

mask_percentage_list = np.load("mask_percentage_list.npy")
psnr = np.load("psnr.npy")
ssim = np.load("ssim.npy")
plt.figure(figsize=(20, 12))
plt.plot(mask_percentage_list, ssim, 'o', markersize=2, label='SSIM')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("SSIM")
plt.xlabel("Mask Percentage")
plt.ylabel("Value")
plt.legend(loc='upper right')
plt.savefig("ssim.png")
plt.close()
plt.figure(figsize=(20, 12))
# plt.plot(mask_percentage_list, psnr, 'o', label='PSNR')
plt.plot(mask_percentage_list, psnr, 'o', markersize=2, label='PSNR')
# Grid
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("PSNR")
# X and Y axis title
plt.xlabel("Mask Percentage")
plt.ylabel("Value")
plt.legend(loc='upper right')
plt.savefig("psnr.png")
plt.close()




# Find the position of best PSNR and SSIM in the list
psnr_max = np.argmax(psnr)
ssim_max = np.argmax(ssim)
print(psnr_max)
print(ssim_max)
print("Best PSNR: ", mask_percentage_list[psnr_max], psnr[psnr_max])
print("Best SSIM: ", mask_percentage_list[ssim_max], ssim[ssim_max])