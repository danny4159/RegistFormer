from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import h5py
from src import utils
import os
from torch.utils.data import Dataset
from monai.transforms import RandFlipd, RandRotate90d, Compose, RandCropd
from monai.utils.type_conversion import convert_to_tensor
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as nnF

# Misalign
import torchio as tio
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from monai.transforms import Affine
import random
import math

def padding_height_width(tensorA, tensorB, tensorC=None, tensorD=None, target_size=(256, 256), pad_value=-1):
    # 기존 코드와 동일
    # Determine if the input is 2D or 3D based on the number of dimensions
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
        assert h >= target_size[0] and w >= target_size[1], "Input tensor size must be larger than min_size"
        assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"
        if tensorC is not None:
            assert len(tensorC.shape) == 3, "Input tensor C must have 3 dimensions (C, H, W)"
        if tensorD is not None:
            assert len(tensorD.shape) == 3, "Input tensor D must have 3 dimensions (C, H, W)"
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, d = tensorA.shape
        assert h >= target_size[0] and w >= target_size[1], "Input tensor size must be larger than min_size"
        assert len(tensorB.shape) == 4, "Input tensor B must have 4 dimensions (C, H, W, D)"
        if tensorC is not None:
            assert len(tensorC.shape) == 4, "Input tensor C must have 4 dimensions (C, H, W, D)"
        if tensorD is not None:
            assert len(tensorD.shape) == 4, "Input tensor D must have 4 dimensions (C, H, W, D)"
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    # Calculate padding
    pad_top = (target_size[0] - h + 1) // 2 if h < target_size[0] else 0
    pad_bottom = (target_size[0] - h) // 2 if h < target_size[0] else 0
    pad_left = (target_size[1] - w + 1) // 2 if w < target_size[1] else 0
    pad_right = (target_size[1] - w) // 2 if w < target_size[1] else 0

    # Apply padding
    if pad_top != 0 or pad_left != 0:
        tensorA = nnF.pad(tensorA, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        tensorB = nnF.pad(tensorB, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        if tensorC is not None:
            tensorC = nnF.pad(tensorC, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        if tensorD is not None:
            tensorD = nnF.pad(tensorD, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB

def random_crop_height_width(tensorA, tensorB, tensorC=None, tensorD=None, target_size=(128, 128)):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Determine if the input is 2D or 3D based on the number of dimensions
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
        assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"
        if tensorC is not None:
            assert len(tensorC.shape) == 3, "Input tensor C must have 3 dimensions (C, H, W)"
        if tensorD is not None:
            assert len(tensorD.shape) == 3, "Input tensor D must have 3 dimensions (C, H, W)"
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, d = tensorA.shape
        assert len(tensorB.shape) == 4, "Input tensor B must have 4 dimensions (C, H, W, D)"
        if tensorC is not None:
            assert len(tensorC.shape) == 4, "Input tensor C must have 4 dimensions (C, H, W, D)"
        if tensorD is not None:
            assert len(tensorD.shape) == 4, "Input tensor D must have 4 dimensions (C, H, W, D)"
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    # Calculate the top left corner of the random crop
    top = (
        torch.randint(0, h - target_size[0] + 1, size=(1,)).item()
        if h > target_size[0]
        else 0
    )
    left = (
        torch.randint(0, w - target_size[1] + 1, size=(1,)).item()
        if w > target_size[1]
        else 0
    )

    if len(tensorA.shape) == 3:
        # Perform the crop for 2D tensors
        tensorA = F.crop(tensorA, top, left, target_size[0], target_size[1])
        tensorB = F.crop(tensorB, top, left, target_size[0], target_size[1])
        if tensorC is not None:
            tensorC = F.crop(tensorC, top, left, target_size[0], target_size[1])
        if tensorD is not None:
            tensorD = F.crop(tensorD, top, left, target_size[0], target_size[1])
    elif len(tensorA.shape) == 4:
        # Perform the crop for 3D tensors
        tensorA = tensorA[:, top:top + target_size[0], left:left + target_size[1], :]
        tensorB = tensorB[:, top:top + target_size[0], left:left + target_size[1], :]
        if tensorC is not None:
            tensorC = tensorC[:, top:top + target_size[0], left:left + target_size[1], :]
        if tensorD is not None:
            tensorD = tensorD[:, top:top + target_size[0], left:left + target_size[1], :]

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB
    
def even_crop_height_width(tensorA, tensorB, tensorC=None, tensorD=None, multiple=(16, 16)):
    """
    Crop the image to the target size evenly from all sides.

    Args:
        tensorA (Tensor): Image to be cropped.
        tensorB (Tensor): Second Image to be cropped.
        target_size (tuple): Desired output size (height, width).
        tensorC (Tensor, optional): Third image to be cropped.
        tensorD (Tensor, optional): Fourth image to be cropped.

    Returns:
        Tensor: Cropped images.
    """
    if len(tensorA.shape) == 3:
        # 2D case
        _, h, w = tensorA.shape
    elif len(tensorA.shape) == 4:
        # 3D case
        _, h, w, _ = tensorA.shape
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions")

    new_h = (h // multiple[0]) * multiple[0]
    new_w = (w // multiple[1]) * multiple[1]

    # Calculate cropping dimensions
    top = (h - new_h) // 2
    bottom = h - new_h - top
    left = (w - new_w) // 2
    right = w - new_w - left

    if len(tensorA.shape) == 3:
        # Crop images
        tensorA = F.crop(tensorA, top, left, new_h, new_w)
        tensorB = F.crop(tensorB, top, left, new_h, new_w)
        if tensorC is not None:
            tensorC = F.crop(tensorC, top, left, new_h, new_w)
        if tensorD is not None:
            tensorD = F.crop(tensorD, top, left, new_h, new_w)
    elif len(tensorA.shape) == 4:
        # Crop images
        tensorA = tensorA[:, top:top + new_h, left:left + new_w, :]
        tensorB = tensorB[:, top:top + new_h, left:left + new_w, :]
        if tensorC is not None:
            tensorC = tensorC[:, top:top + new_h, left:left + new_w, :]
        if tensorD is not None:
            tensorD = tensorD[:, top:top + new_h, left:left + new_w, :]

    if tensorC is not None and tensorD is not None:
        return tensorA, tensorB, tensorC, tensorD
    elif tensorC is not None:
        return tensorA, tensorB, tensorC
    else:
        return tensorA, tensorB


log = utils.get_pylogger(__name__)

class dataset_SynthRAD(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: Optional[str] = None,
        is_3d: bool = False,
        padding_size: Optional[Tuple[int, int]] = None,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        crop_size: Optional[Tuple[int, int]] = None,
        reverse: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.is_3d = is_3d
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.reverse = reverse

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        self.patient_keys = []

        if self.is_3d:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())

            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )
        else:
            with h5py.File(self.data_dir, "r") as file:
                self.patient_keys = list(file[self.data_group_1].keys())
                self.slice_counts = [file[self.data_group_1][key].shape[-1] for key in self.patient_keys]
                self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=flip_prob, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"] if not self.data_group_3 else ["A", "B", "C"], prob=rot_prob, spatial_axes=[0, 1]),
                ]
            )

    def __len__(self):
        if self.is_3d:
            return len(self.patient_keys)
        else:
            return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        if self.is_3d:
            patient_key = self.patient_keys[idx]
            with h5py.File(self.data_dir, "r") as file:
                A = file[self.data_group_1][patient_key][...]
                B = file[self.data_group_2][patient_key][...]
                if self.data_group_3:
                    C = file[self.data_group_3][patient_key][...]
        else:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            patient_key = self.patient_keys[patient_idx]
            with h5py.File(self.data_dir, "r") as file:
                A = file[self.data_group_1][patient_key][..., slice_idx]
                B = file[self.data_group_2][patient_key][..., slice_idx]
                if self.data_group_3:
                    C = file[self.data_group_3][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()
        if self.data_group_3:
            C = torch.from_numpy(C).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}
        if self.data_group_3:
            data_dict["C"] = C

        A = data_dict["A"]
        B = data_dict["B"]
        if self.data_group_3:
            C = data_dict["C"]

        if self.padding_size:
            if self.data_group_3:
                A, B, C = padding_height_width(A, B, C, target_size=self.padding_size)
            else:
                A, B = padding_height_width(A, B, target_size=self.padding_size)

        data_dict = self.aug_func(data_dict)

        if self.crop_size:
            if self.data_group_3:
                A, B, C = random_crop_height_width(A, B, C, target_size=self.crop_size) 
            else:
                A, B = random_crop_height_width(A, B, target_size=self.crop_size) 
        else:
            if self.data_group_3:
                A, B, C = even_crop_height_width(A, B, C, multiple=(16, 16)) # 16의 배수로 Crop
            else:
                A, B = even_crop_height_width(A, B, multiple=(16, 16)) # 16의 배수로 Crop
                
        if self.reverse:
            if self.data_group_3:
                return C, B, A
            else:
                return B, A
        else:
            if self.data_group_3:
                return A, B, C
            else:
                return A, B

    def get_patient_slice_idx(self, idx):
        if self.is_3d:
            patient_idx = np.searchsorted(self.cumulative_slice_counts, idx + 1) - 1
            slice_idx = idx - self.cumulative_slice_counts[patient_idx]
            return patient_idx, slice_idx
        else:
            return idx, None

class dataset_Histological(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_group_1: str,
        data_group_2: str,
        data_group_3: str,
        data_group_4: str,
        padding_size: Optional[Tuple[int, int]] = None,
        flip_prob: float = 0.0,
        rot_prob: float = 0.0,
        crop_size: Optional[Tuple[int, int]] = None,
        reverse: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_group_1 = data_group_1
        self.data_group_2 = data_group_2
        self.data_group_3 = data_group_3
        self.data_group_4 = data_group_4
        self.padding_size = padding_size
        self.crop_size = crop_size
        self.reverse = reverse

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        with h5py.File(self.data_dir, "r") as file:
            self.patient_keys = list(file[self.data_group_1].keys())

        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B", "C" ,"D"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B", "C", "D"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

    def __len__(self):
        return len(self.patient_keys)

    def __getitem__(self, idx):
        patient_key = self.patient_keys[idx]
        with h5py.File(self.data_dir, "r") as file:
            A = file[self.data_group_1][patient_key][...]
            B = file[self.data_group_2][patient_key][...]
            C = file[self.data_group_3][patient_key][...]
            D = file[self.data_group_4][patient_key][...]

        # 2D 이미지는 unsqueeze로 채널 추가
        A = torch.from_numpy(A).unsqueeze(0).float()  # (H, W) -> (1, H, W)
        B = torch.from_numpy(B).unsqueeze(0).float()  # (H, W) -> (1, H, W)
        C = torch.from_numpy(C).permute(2, 0, 1).float()  # RGB 이미지 채널을 맨 앞으로 이동 (H, W, 3) -> (3, H, W)
        D = torch.from_numpy(D).permute(2, 0, 1).float()  # RGB 이미지 채널을 맨 앞으로 이동 (H, W, 3) -> (3, H, W)

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B, "C": C, "D": D}

        if self.padding_size:
            A, B, C, D = padding_height_width(A, B, C, D, target_size=self.padding_size)

        data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]
        D = data_dict["D"]

        if self.crop_size:
            A, B, C, D = random_crop_height_width(A, B, C, D, target_size=self.crop_size) 
        else:
            A, B, C, D = even_crop_height_width(A, B, C, D) 

        if self.reverse:
            return C, D, A, B
        else:
            return A, B, C, D
        


def download_process_MR_3T_7T(data_dir: str,
                         write_dir: str,
                         misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
                         misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
                         degree: float = 0.0,      # maximum rotation in z direction (float)
                         motion_prob: float = 0.0, # the probability of occurrence of motion.
                         deform_prob: float = 0.0, # the probability of performing deformation.
                         ret: bool = False
                        ):
    """Downloads, preprocesses and saves the IXI dataset. The images are randomly translated.

    Args:
        data_dir (str): The directory where the input dataset is located.
        write_dir (str): The directory where the processed dataset will be saved.
        misalign_x (float): Maximum allowable misalignment along the x dimension. Defaults to 0.0.
        misalign_y (float): Maximum allowable misalignment along the y dimension. Defaults to 0.0.
        degree (float): Maximum rotation in z direction. Defaults to 0.0.
        motion_prob (float): The probability of occurrence of motion. Defaults to 0.0.
        deform_prob (float): The probability of performing deformation. Defaults to 0.0.
        ret (bool): If True, the function also returns the processed dataset. Defaults to False.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: A pair of tensors representing the processed datasets
        for A and B. This is returned only when 'ret' is set to True.
    """

    with h5py.File(data_dir, "r") as f:
        data_A = np.array(f["data_x"]) # H, W, S
        data_B = np.array(f["data_y"])

    # create torch tensors
    data_A = np.transpose(data_A, (2, 0, 1)) # H, W, Slice -> # Slice, H, W
    # data_A = np.flipud(data_A)
    data_A = np.expand_dims(data_A, axis=0) # 1, Slice, H, W
    print('data_A shape: ', data_A.shape)

    data_B = np.transpose(data_B, (2, 0, 1))
    # data_B = np.flipud(data_B)
    data_B = np.expand_dims(data_B, axis=0)
    print('data_B shape: ', data_B.shape)

    # ensure that data is in range [-1,1]
    data_A[data_A < 0] = 0
    data_B[data_B < 0] = 0

    data_A, data_B = Deformation(data_A.copy(), data_B.copy(), deform_prob)
    data_A, data_B = Motion_artifacts(data_A, data_B, motion_prob)

    data_A = (data_A - 0.5) / 0.5
    data_B = (data_B - 0.5) / 0.5

    log.info(f"Preparing the misalignment from <{data_dir}>")

    data_A, data_B = Rotate_images(data_A, data_B, degree)

    num_slices_per_patient = 200
    num_patients = data_A.shape[1] // num_slices_per_patient

    log.info(f"Saving the prepared dataset to <{write_dir}>")

    with h5py.File(write_dir, "w") as hw:
        data_A_group = hw.create_group("3T")
        data_B_group = hw.create_group("7T")

        for patient_idx in range(num_patients):
            start_idx = patient_idx * num_slices_per_patient
            end_idx = start_idx + num_slices_per_patient

            patient_data_A_list = []
            patient_data_B_list = []

            for sl in range(start_idx, end_idx): # Translation each slice
                A = torch.from_numpy(data_A[:, sl, :, :])
                B = torch.from_numpy(data_B[:, sl, :, :])

                A, B = Translate_images(A, B, misalign_x, misalign_y)

                patient_data_A_list.append(A)
                patient_data_B_list.append(B)

            patient_data_A = torch.stack(patient_data_A_list) # num_slices, 1, H, W
            patient_data_B = torch.stack(patient_data_B_list)

            patient_data_A = patient_data_A.squeeze(1) # num_slices, H, W
            patient_data_B = patient_data_B.squeeze(1)

            patient_data_A = patient_data_A.permute(1, 2, 0) # H, W, num_slices
            patient_data_B = patient_data_B.permute(1, 2, 0)

            data_A_group.create_dataset(str(patient_idx + 1), data=patient_data_A.numpy())
            data_B_group.create_dataset(str(patient_idx + 1), data=patient_data_B.numpy())

    if ret:
        return patient_data_A, patient_data_B
    else:
        return

################################################################################################################################
################################################################################################################################

def Deformation(A:np.ndarray,
                    B:np.ndarray,
                    deform_prob:float):
    """Apply dense random elastic deformation.

    Args:
        A (np.ndarray: An input image ndarray.
        deform_prob (float): The probability of performing deformation.

    Returns:
        Tuple[np.ndarray]: A deformed image.
    """

    elastic_transform = tio.transforms.RandomElasticDeformation( ## Tuning parameters while observing the results
        num_control_points=6,  # Number of control points along each dimension.
        max_displacement=(30,30,0),    # Maximum displacement along each dimension at each control point.
    )

    elastic_A = elastic_transform(A)
    elastic_B = elastic_transform(B)
    num_samples = A.shape[1]
    num_transform_samples = int(num_samples * deform_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A:
        A[:,index,:,:] = elastic_A[:,index,:,:]

    for index in indices_B:
        B[:,index,:,:] = elastic_B[:,index,:,:]

    return A, B

def Motion_region(raw_img, motion_img, prob):
    raw_k = fftshift(fftn(raw_img))
    motion_k = fftshift(fftn(motion_img))
    diff = math.ceil(2.56*prob)

    raw_k[:,128-diff : 128,:] = motion_k[:,128-diff : 128,:]

    res = abs(ifftn(ifftshift(raw_k)))
    return res

def Motion_artifacts(A:np.ndarray,
                         B:np.ndarray,
                         motion_prob:float):
    """Simulates motion artifacts.

    Args:
        A (np.ndarray): An input image ndarray.
        motion_prob (float): The probability of occurrence of motion.

    Returns:
        Tuple[np.ndarray]: A motion artifacts-injected tensor image.
    """

    # Define the 3D-RandomMotion transform
    random_motion = tio.RandomMotion(
        degrees=(5,5),              # Maximum rotation angle in degrees
        translation=(10,10),         # Maximum translation in mm
        num_transforms=10         # Number of motion transformations to apply
    )

    A = np.transpose(A, [0,2,3,1])
    B = np.transpose(B, [0,2,3,1])
    motion_A = random_motion(A)
    motion_A_prob = Motion_region(A, motion_A, prob=6)
    motion_B = random_motion(B)
    motion_B_prob = Motion_region(B, motion_B, prob=6)

    num_samples = A.shape[3]
    num_transform_samples = int(num_samples * motion_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A:
        A[:,:,:,index] = motion_A_prob[:,:,:,index]

    for index in indices_B:
        B[:,:,:,index] = motion_B_prob[:,:,:,index]

    A = np.transpose(A, [0,3,1,2])
    B = np.transpose(B, [0,3,1,2])

    return A, B

def Rotate_images(A:np.ndarray,
                      B:np.ndarray,
                      degree:float):
    """Rotates a given image (A) by random degree along z dimensions.

    Args:
        A (np.ndarray): An input image ndarray.
        degree (float): Maximum allowable degree along the z dimension.

    Returns:
        Tuple[np.ndarray]: A rotated tensor image.
    """

    # rotation
    transform = tio.RandomAffine(
        scales=0,
        degrees=(degree,0,0),        # z-axis 3D-Rotation range in degrees.
        translation=(0,0,0),
        default_pad_value='otsu',  # edge control, fill value is the mean of the values at the border that lie under an Otsu threshold.
    )

    rotated_A = transform(A)
    rotated_B = transform(B)
    
    return rotated_A, rotated_B

def Translate_images(A: torch.Tensor,
              B: torch.Tensor,
              misalign_x: float,
              misalign_y: float):
    """Translates two given images (A and B) by random misalignments along x and y dimensions.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        misalign_x (float): Maximum allowable misalignment along the x dimension.
        misalign_y (float): Maximum allowable misalignment along the y dimension.

    Returns:
        Tuple[torch.Tensor]: A pair of tensors representing the translated images.
    """

    # translation (changed : misalign is the smae but magnitude is different)
    _misalign_x = np.random.uniform(-1, 1, size=2)
    _misalign_y = np.random.uniform(-1, 1, size=2)

    misalign_x = misalign_x * _misalign_x
    misalign_y = misalign_y * _misalign_y

    # misalign_x = np.random.uniform(-misalign_x, misalign_x, size=2) # This is the previous version
    # misalign_y = np.random.uniform(-misalign_y, misalign_y, size=2)

    translate_params_A = (misalign_y[0], misalign_x[0])  # (y, x) for image A
    translate_params_B = (misalign_y[1], misalign_x[1])  # (y, x) for image B

    # create affine transform
    affine_A = Affine(translate_params=translate_params_A)
    affine_B = Affine(translate_params=translate_params_B)

    moved_A = affine_A(A)
    moved_B = affine_B(B)

    return moved_A[0], moved_B[0]