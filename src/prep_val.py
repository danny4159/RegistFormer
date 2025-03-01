import numpy as np
import nibabel as nib
import h5py
import os
import tqdm

# skull-strip: docker run -v /home/kanghyun/misalign-benchmark/data:/home/kanghyun/misalign-benchmark/data docker.io/freesurfer/synthstrip -i /home/kanghyun/misalign-benchmark/data/IXI/val/nii/a_1.nii.gz -m /home/kanghyun/misalign-benchmark/data/IXI/val/nii/a_1_mask.nii.gz


def change_numpy_nii(a, b):
    assert (
        a.ndim == b.ndim == 3
    ), "All input arrays must have the same number of dimensions (3)"

    # scale to [0, 1] and [0, 255]
    a, b = (((a + 1) / 2) * 255, ((b + 1) / 2) * 255)

    # type to np.int16
    a, b = (a.astype(np.float32), b.astype(np.float32))

    # transpose 1, 2 dim (for viewing on ITK-SNAP)
    a, b = (
        np.transpose(a, axes=(1, 0, 2))[:, ::-1],
        np.transpose(b, axes=(1, 0, 2))[:, ::-1],
    )

    affine_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # Create Nifti1Image for each
    a_nii, b_nii = (
        nib.Nifti1Image(a, affine_matrix),
        nib.Nifti1Image(b, affine_matrix),
    )

    return a_nii, b_nii


def save_nii(a_nii, b_nii, subject_number, folder_path):
    nib.save(a_nii, os.path.join(folder_path, f"a_{subject_number}.nii.gz"))
    nib.save(b_nii, os.path.join(folder_path, f"b_{subject_number}.nii.gz"))
    return


print("For val data")

# Load data
with h5py.File("/home/kanghyun/misalign-benchmark/data/IXI/val/data.mat", "r") as f:
    data_A = np.array(f["data_x"][:, :, :, 1])
    data_B = np.array(f["data_y"][:, :, :, 1])

img_A = []
img_B = []


subject_number = 1
folder_path = "/home/kanghyun/misalign-benchmark/data/IXI/val/nii"
os.makedirs(folder_path, exist_ok=True)

for sl in tqdm.tqdm(range(data_A.shape[2]), desc="Saving nii"):
    img_A.append(data_A[:, :, sl])
    img_B.append(data_B[:, :, sl])

    if len(img_A) == 91:
        a_nii = np.stack(img_A, -1)
        b_nii = np.stack(img_B, -1)

        a_nii, b_nii = change_numpy_nii(a_nii, b_nii)
        save_nii(a_nii, b_nii, subject_number, folder_path)
        subject_number += 1
        img_A = []
        img_B = []

print("For test data")

with h5py.File("/home/kanghyun/misalign-benchmark/data/IXI/test/data.mat", "r") as f:
    data_A = np.array(f["data_x"][:, :, :, 1])
    data_B = np.array(f["data_y"][:, :, :, 1])

img_A = []
img_B = []

print(data_A.shape)
subject_number = 1
folder_path = "/home/kanghyun/misalign-benchmark/data/IXI/test/nii"
os.makedirs(folder_path, exist_ok=True)

for sl in tqdm.tqdm(range(data_A.shape[2]), desc="Saving nii"):
    img_A.append(data_A[:, :, sl])
    img_B.append(data_B[:, :, sl])

    if len(img_A) == 91:
        a_nii = np.stack(img_A, -1)
        b_nii = np.stack(img_B, -1)

        a_nii, b_nii = change_numpy_nii(a_nii, b_nii)
        save_nii(a_nii, b_nii, subject_number, folder_path)
        subject_number += 1
        img_A = []
        img_B = []
