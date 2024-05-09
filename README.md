# Generative Image Inpainting

An open source framework for generative image inpainting task, with the support of [Contextual Attention](https://arxiv.org/abs/1801.07892) (CVPR 2018) and [Gated Convolution](https://arxiv.org/abs/1806.03589) (ICCV 2019 Oral).

## Run

0. Requirements:

   * Install python3.
   * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
   * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`).
   * Dataset directory `./imagenet`, original from [imagenet (kaggle.com)](https://www.kaggle.com/datasets/lijiyu/imagenet)
1. Training:

   * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)), using `python flist_generate.py`
   * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
   * Run `python train.py`.
2. Resume training:

   * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
   * Run `python train.py`.
3. Testing:

* Run `python test.py`. Setting up the start and end indexes

4. Evaluation:

* Run `python plot.py`.

## Pretrained models

[Places2](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO?usp=sharing)

Download the model dirs and put it under `model_logs/` (rename `checkpoint.txt` to `checkpoint` because google drive automatically add ext after download). Run testing or resume training as described above. All models are trained with images of resolution 256x256 and largest hole size 128x128, above which the results may be deteriorated. We provide several example test cases. Please run:

```bash
python test.py
```

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

## Citing

```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
