# RegistFormer (Breaking dilemma of cross-modality image registration)

# [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10944116)
Improving Pelvic MR-CT Image Alignment with Self-supervised Reference-Augmented Pseudo-CT Generation Framework, WACV 2024
Daniel Kim, Mohammed A. Al-masni, Jaehun Lee, Dong-Hyun Kim, Kanghyun Ryu
2024 WACV (Winter Conference on Applications of Computer Vision)

![Comparison with existing Registration method](img/MR-CT_Registraion_Comparison(WACV).png)


![Overall architecture of Proposed method](img/MR-CT_Registraion_Comparison(WACV).png)


 
## 1. Code template

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking


## 2. Library installation

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
The preprocessed dataset required for training, validation, and testing is shared at ['this link'](https://drive.google.com/drive/folders/1Y9tr9mZ58avHRubUlUpDbpTyVpwTu9ck?usp=sharing) for convenience.
Put it in the following path: `/data`.
Details of the preprocessing steps are described in the paper.

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
python src/train.py model='registformer.yaml' trainer.devices=[0] tags='Registformer_MrCtPelvis_MUNIT_Test' data.use_split_inference=true train=False ckpt_path='<YOUR_PROJECT_ROOT>/pretrained/proposed/proposed_weight.ckpt'
# data.use_split_inference: Whether to split the image into two parts for inference. Set it to 'false' if the memory is sufficient.
```  

## 6. How to train

### 1st. Stage 1
```bash
python src/train.py model='munit.yaml' tags='MUNIT_Train' trainer.devices=[0] data.train_file=Ver3_AllPatientSameSize_final_2.h5 data.val_file=Ver3_AllPatientSameSize_final_2.h5 data.test_file=Ver3_AllPatientSameSize_final_2.h5
```

### 2nd. Registration
Before running the Registration step, you must perform inference on the train, validation, and test sets using the weights trained in Stage 1. The outputs of this inference should then be preprocessed into a new HDF5 file. This new file will be used as the 'Registration_MR_CT_synCT.h5'.
```bash
python src/train.py model='voxelmorph_original.yaml' tags='Voxelmorph_Original_CTsynCTPelvis_2D_Train' trainer.devices=[2] data.batch_size=1 data.data_group_3='syn_CT' data.train_file=Registration_MR_CT_synCT.h5 data.val_file=Registration_MR_CT_synCT.h5 data.test_file=Registration_MR_CT_synCT.h5 model.params.lambda_grad=0 model.params.lambda_mask_l2=0 model.params.lambda_smooth=0.5 model.netR_A.inshape=[384,320] data.crop_size=null
```  

### 3rd. Stage 2
Stage 2 is trained using the weights obtained from Stage 1 and Registration.
```bash
python src/train.py model='registformer.yaml' trainer.devices=[0] tags='Registformer_MrCtPelvis_MUNIT_Train' data.use_split_inference=true data.train_file=Ver3_AllPatientSameSize_final_2.h5 data.val_file=Ver3_AllPatientSameSize_final_2.h5 data.test_file=Ver3_AllPatientSameSize_final_2.h5 model.netG_A.synth_path=pretrained/synthesis/munit_synthesis_epoch98.ckpt model.netG_A.regist_path='pretrained/registration/Voxelmorph_2D_CT_SynCT.ckpt model.netG_A.regist_size=[384,320]
# data.use_split_inference: Whether to split the image into two parts for inference. Set it to 'false' if the memory is sufficient.
# model.netG_A.synth_path: Path to the pretrained weight file for Stage 1 (synthesis network).
# model.netG_A.regist_path: Path to the pretrained weight file for the registration network.
# model.netG_A.regist_size: Height and width of the preprocessed dataset.
```
