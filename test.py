import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   
from inpaint_model import InpaintCAModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='data/val', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='data/val_mask', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='data/val_output', type=str,
                    help='Where to write output.')
parser.add_argument('--grouth_truth', default='data/val_image_grouth_truth', type=str,
                    help='Where to read grouth_truth.')
parser.add_argument('--checkpoint_dir', default='model_logs/release_imagenet_256', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()
    psnr_list = []
    ssim_list = []
    start = int(input('Start index: '))
    end = int(input('End index: '))
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        model = InpaintCAModel()
        for i in range(start, end):
            # Make the output directory
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            image = cv2.imread(f"{args.image}/{i}.jpg")
            mask = cv2.imread(f"{args.mask}/{i}.jpg")
            grouth_truth = cv2.imread(f"{args.grouth_truth}/{i}.jpg")
            assert image.shape == mask.shape
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            result = sess.run(output)
            # Write grouth_truth for debugging
            psnr_value = psnr(result[0][:, :, ::-1], grouth_truth)
            ssim_value = ssim(result[0][:, :, ::-1], grouth_truth, multichannel=True)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

            
            # Plot the images and save the output as 4 images, original, mask, output, grouth_truth in a subplot
            plt.figure(figsize=(15, 5))
            plt.subplot(141)
            plt.imshow(cv2.imread(f"{args.image}/{i}.jpg")[:, :, ::-1])
            plt.title("Original")
            plt.axis("off")
            plt.subplot(142)
            plt.imshow(cv2.imread(f"{args.mask}/{i}.jpg")[:, :, ::-1])
            plt.title("Mask")
            plt.axis("off")
            plt.subplot(143)
            plt.imshow(result[0])
            plt.title("Output")
            plt.axis("off")
            plt.subplot(144)
            plt.imshow(grouth_truth[:, :, ::-1])
            plt.title("Grouth Truth")
            plt.axis("off")
            plt.suptitle(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")
            plt.tight_layout()
            plt.savefig(f"{args.output}/{i}.jpg")
            plt.close()
    
    # Save the psnr_list and ssim_list into 2 files for visualization
    np.save(f"psnr_from_{start}_to_{end-1}.npy", np.array(psnr_list))
    np.save(f"ssim.npy_{start}_to_{end-1}.npy", np.array(ssim_list))
            
