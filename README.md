# RegistFormer

## 1. Description

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking


## 2. Installation

```bash
conda env create -f environment.yml
```

If you encounter an error while setting up the environment using the environment.yaml file, please install the libraries in the following order:
```bash
conda create --name RbG_framework python=3.9.16
conda activate RbG_framework
pip install hydra-core
python -m pip install lightning
pip install pyrootutils
pip install rich
pip install hydra-colorlog
pip install h5py
pip install monai-weekly
pip install torchvision
pip install torchio
pip install lpips
pip install opencv-python
pip install torch-fidelity
pip install gdown
pip install tensorboardX
pip install -U openmim
mim install mmcv
```  

## 3. Dataset
### Dataset 
Grand challenge ['SynthRAD 2023'](https://synthrad2023.grand-challenge.org/) Pelvis MR, CT.
The demo dataset for testing can be downloaded from ['this link'](https://drive.google.com/drive/folders/1Vvm4NtNGuHSkscJDLh9Kgs0ssupe4EZ2?usp=sharing/).
Put it in the following path: `/data`.

### Dataset Structure (HDF5 Format: Hierarchical Data Structure)

The dataset is stored in an HDF5 file with the following hierarchical structure:

- **Group**: MR
  - **Dataset**: `1PA001`, `1PA004`, ... (Patient number)

- **Group**: CT
  - **Dataset**: `1PA001`, `1PA004`, ... (Patient number)

All datasets must be resized to have the same width and height dimensions.

## 4. Pretrained weight 
Pretrained weights can be downloaded from [this link](https://drive.google.com/drive/folders/1dR1kGKsZQCLMtXnNqJ8Arm5aFl2IslrX?usp=sharing/).  
Put it in the following path: `/pretrained`.  

## 5. How to test (with pretrained weight)

### Stage 1
```bash
python src/train.py model='munit.yaml' tags='MUNIT_Test' trainer.devices=[0] train=False ckpt_path='<YOUR_PROJECT_PATH>/pretrained/synthesis/munit_synthesis_epoch98.ckpt'
# <YOUR_PROJECT_ROOT>: Put your project path (e.g., /SSD_8TB/RbG_framework).
```  

### Stage 2
```bash
python src/train.py model='registformer.yaml' trainer.devices=[0] tags='Registformer_MrCtPelvis_MUNIT_Test' data.use_split_inference=false train=False ckpt_path='<YOUR_PROJECT_ROOT>/pretrained/proposed/proposed_weight.ckpt'
# data.use_split_inference: Whether to split the image into two parts for inference. Set it to 'true' if the memory is sufficient.
```  

## 6. How to train

### 1st. Stage 1
```bash
python src/train.py model='munit.yaml' tags='MUNIT_Train' trainer.devices=[0] data.train_file=<PREPROCESSED_DATASET>.h5 data.val_file=<PREPROCESSED_DATASET>.h5 data.test_file=<PREPROCESSED_DATASET>.h5
# <PREPROCESSED_DATASET>: Use your preprocessed dataset following the above Dataset Structure.
```

### 2nd. Registration
Before running the Registration step, you must perform inference on the train, validation, and test sets using the weights trained in Stage 1. The outputs of this inference should then be preprocessed into a new HDF5 file. This new file will be used as the <PREPROCESSED_DATASET_FOR_REGISTRATION>.
```bash
python src/train.py model='voxelmorph_original.yaml' tags='Voxelmorph_Original_CTsynCTPelvis_2D_Train' trainer.devices=[2] data.batch_size=1 data.data_group_3='syn_CT' data.train_file='<PREPROCESSED_DATASET_FOR_REGISTRATION>.h5' data.val_file='<PREPROCESSED_DATASET_FOR_REGISTRATION>.h5' data.test_file='<PREPROCESSED_DATASET_FOR_REGISTRATION>.h5' model.netR_A.inshape=[384,320]
# <PREPROCESSED_DATASET_FOR_REGISTRATION>: The dataset generated by preprocessing the inference results from Stage 1.
```  

### 3rd. Stage 2
Stage 2 is trained using the weights obtained from Stage 1 and Registration.
```bash
python src/train.py model='RbG.yaml' trainer.devices=[0] tags='Registformer_MrCtPelvis_MUNIT_Train' data.use_split_inference=false data.train_file='<PREPROCESSED_DATASET>.h5' data.val_file='<PREPROCESSED_DATASET>.h5' data.test_file='<PREPROCESSED_DATASET>.h5' model.netG_A.synth_path='pretrained/MR-CT/stage1_synthesis/<PRETRAINED_WEIGHT_STAGE1>.ckpt' model.netG_A.regist_path='pretrained/MR-CT/registration/<PRETRAINED_WEIGHT_REGISTRATION>' model.netG_A.regist_size=[<HEIGHT,WIDTH>]
# data.use_split_inference: Whether to split the image into two parts for inference. Set it to 'true' if the memory is sufficient.
# <PRETRAINED_WEIGHT_STAGE1>: Put the filename of the pretrained weight for stage 1.
# <PRETRAINED_WEIGHT_REGISTRATION>: Put the filename of the pretrained weight for registration network.
# <HEIGHT,WIDTH>: Put the height and width size of preprocessed dataset.
```
