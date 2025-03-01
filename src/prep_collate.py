from pathlib import Path
import nibabel as nib
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from scipy.ndimage import binary_dilation


def calc_mutual_information(a, b):
    hist_2d, _, _ = np.histogram2d(a, b, bins=30)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# Define the paths
path1 = Path("/home/kanghyun/misalign-benchmark/data/IXI/val/nii")
path2 = Path("/home/kanghyun/misalign-benchmark/data/IXI/test/nii")

# Concatenate paths
paths = [path1, path2]

a_arrays, b_arrays, a_mask_arrays, b_mask_arrays = [], [], [], []

for path in paths:
    # Gather all .nii.gz files starting with 'a'
    all_files = list(path.glob("**/a*.nii.gz"))

    # Filter out files that contain 'mask'
    filtered_files = [f for f in all_files if "mask" not in str(f)]

    # Sort the list
    a_sorted_files = sorted(
        filtered_files, key=lambda f: int(str(f).split("_")[1].split(".nii.gz")[0])
    )

    # Repeat the process for the 'b' files
    all_files = list(path.glob("**/b*.nii.gz"))
    filtered_files = [f for f in all_files if "mask" not in str(f)]
    b_sorted_files = sorted(
        filtered_files, key=lambda f: int(str(f).split("_")[1].split(".nii.gz")[0])
    )

    # Repeat the process for the 'a_mask' files
    all_files = list(path.glob("**/a*mask.nii.gz"))
    a_mask_sorted_files = sorted(
        all_files, key=lambda f: int(str(f).split("_")[1].split("mask.nii.gz")[0])
    )

    # Repeat the process for the 'b_mask' files
    all_files = list(path.glob("**/b*mask.nii.gz"))
    b_mask_sorted_files = sorted(
        all_files, key=lambda f: int(str(f).split("_")[1].split("mask.nii.gz")[0])
    )

    # Now you can loop over sorted_files by pairs:
    for a_file, b_file, a_mask_file, b_mask_file in zip(
        a_sorted_files, b_sorted_files, a_mask_sorted_files, b_mask_sorted_files
    ):
        # Extract the numerical postfixes from the 'a', 'b', 'a_mask', and 'b_mask' filenames
        a_postfix = int(str(a_file).split("_")[1].split(".nii.gz")[0])
        b_postfix = int(str(b_file).split("_")[1].split(".nii.gz")[0])
        a_mask_postfix = int(str(a_mask_file).split("_")[1].split("mask.nii.gz")[0])
        b_mask_postfix = int(str(b_mask_file).split("_")[1].split("mask.nii.gz")[0])

        # Assert that the postfixes match
        assert (
            a_postfix == b_postfix == a_mask_postfix == b_mask_postfix
        ), f"The postfixes do not match for files {a_file}, {b_file}, {a_mask_file}, and {b_mask_file}"

        # Load the files and convert them into numpy arrays
        a_arrays.append(nib.load(a_file).get_fdata())
        b_arrays.append(nib.load(b_file).get_fdata())
        a_mask_arrays.append(nib.load(a_mask_file).get_fdata())
        b_mask_arrays.append(nib.load(b_mask_file).get_fdata())

# Stack the arrays
a_stack = np.stack(a_arrays)
b_stack = np.stack(b_arrays)
a_mask_stack = np.stack(a_mask_arrays)
b_mask_stack = np.stack(b_mask_arrays)

# Create a combined mask where either 'a' or 'b' mask is present
mask_stack = (a_mask_stack + b_mask_stack) > 0

# Initialize the masked stacks
a_masked_stack = np.zeros_like(a_stack)
b_masked_stack = np.zeros_like(b_stack)
# Initialize a list to store mutual information values
mutual_info_values = []

for i in range(a_stack.shape[0]):
    a_masked_stack[i] = a_stack[i] * mask_stack[i]
    b_masked_stack[i] = b_stack[i] * mask_stack[i]

    # Flatten the arrays and remove zero elements (where mask was applied)
    a_flat = a_masked_stack[i][mask_stack[i]].ravel()
    b_flat = b_masked_stack[i][mask_stack[i]].ravel()

    # Calculate mutual information
    mutual_info = calc_mutual_information(a_flat, b_flat)

    # Store mutual information value
    mutual_info_values.append(mutual_info)

# Create a numpy array of mutual info values
mutual_info_array = np.array(mutual_info_values)

# Get the indices of top five mutual information values
top_five_indices = np.argsort(mutual_info_array)[-3:]

# Initialize lists to store top three mutual info data and masks
a_val_arrays = []
b_val_arrays = []
a_val_mask_arrays = []
b_val_mask_arrays = []

# Iterate over top three indices and store the corresponding a and b arrays and masks in the validation set lists
for i in top_five_indices:
    a_val_arrays.append(a_arrays[i])
    b_val_arrays.append(b_arrays[i])
    a_val_mask_arrays.append(a_mask_arrays[i])
    b_val_mask_arrays.append(b_mask_arrays[i])

# Convert lists to numpy arrays
a_val_stack = np.stack(a_val_arrays)
b_val_stack = np.stack(b_val_arrays)
a_val_mask_stack = np.stack(a_val_mask_arrays)
b_val_mask_stack = np.stack(b_val_mask_arrays)

# Create a combined mask where either 'a_val' or 'b_val' mask is present
mask_val_stack = (a_val_mask_stack + b_val_mask_stack) > 0

a_val_stack = np.concatenate((a_val_stack[0], a_val_stack[1], a_val_stack[2]), -1)
b_val_stack = np.concatenate((b_val_stack[0], b_val_stack[1], b_val_stack[2]), -1)
mask_val_stack = np.concatenate(
    (mask_val_stack[0], mask_val_stack[1], mask_val_stack[2]), -1
)
mask_val_stack = binary_dilation(mask_val_stack, iterations=2)
# Get all indices
all_indices = np.arange(len(a_arrays))

# Exclude the validation indices to get the test indices
test_indices = np.setdiff1d(all_indices, top_five_indices)

# Initialize lists to store test data and masks
a_test_arrays = []
b_test_arrays = []
a_test_mask_arrays = []
b_test_mask_arrays = []

# Iterate over test indices and store the corresponding a and b arrays and masks in the test set lists
for i in test_indices:
    a_test_arrays.append(a_arrays[i])
    b_test_arrays.append(b_arrays[i])
    a_test_mask_arrays.append(a_mask_arrays[i])
    b_test_mask_arrays.append(b_mask_arrays[i])

# Convert lists to numpy arrays
a_test_stack = np.stack(a_test_arrays)
b_test_stack = np.stack(b_test_arrays)
a_test_mask_stack = np.stack(a_test_mask_arrays)
b_test_mask_stack = np.stack(b_test_mask_arrays)

# Create a combined mask where either 'a_test' or 'b_test' mask is present
mask_test_stack = (a_test_mask_stack + b_test_mask_stack) > 0

a_test_stack = np.concatenate((a_test_stack), -1)
b_test_stack = np.concatenate((b_test_stack), -1)
mask_test_stack = np.concatenate((mask_test_stack), -1)
mask_test_stack = binary_dilation(mask_test_stack, iterations=2)

# Expand the dimensions of a_val_stack
a_val_stack_expanded = np.expand_dims(a_val_stack, axis=-1)
a_val_stack_expanded = np.repeat(a_val_stack_expanded, 3, axis=-1)

# Expand the dimensions of b_val_stack
b_val_stack_expanded = np.expand_dims(b_val_stack, axis=-1)
b_val_stack_expanded = np.repeat(b_val_stack_expanded, 3, axis=-1)

# Expand the dimensions of mask_val_stack
mask_val_stack_expanded = np.expand_dims(mask_val_stack, axis=-1)
mask_val_stack_expanded = np.repeat(mask_val_stack_expanded, 3, axis=-1)

print(a_val_stack_expanded.shape)  # Should be (3, 256, 256, 273)
print(b_val_stack_expanded.shape)  # Should be (3, 256, 256, 273)
print(mask_val_stack_expanded.shape)  # Should be (3, 256, 256, 273)

# Expand the dimensions of a_test_stack
a_test_stack_expanded = np.expand_dims(a_test_stack, axis=-1)
a_test_stack_expanded = np.repeat(a_test_stack_expanded, 3, axis=-1)

# Expand the dimensions of b_test_stack
b_test_stack_expanded = np.expand_dims(b_test_stack, axis=-1)
b_test_stack_expanded = np.repeat(b_test_stack_expanded, 3, axis=-1)

# Expand the dimensions of mask_test_stack
mask_test_stack_expanded = np.expand_dims(mask_test_stack, axis=-1)
mask_test_stack_expanded = np.repeat(mask_test_stack_expanded, 3, axis=-1)

print(a_test_stack_expanded.shape)  # Should be (256, 256, 1092, 3)
print(b_test_stack_expanded.shape)  # Should be (256, 256, 1092, 3)
print(mask_test_stack_expanded.shape)  # Should be (256, 256, 1092, 3)

import h5py

# Create a new HDF5 file
with h5py.File("data_val.mat", "w") as file:
    # Save data_x (a_test_stack)
    file.create_dataset("data_x", data=a_val_stack_expanded)
    # Save data_y (b_test_stack)
    file.create_dataset("data_y", data=b_val_stack_expanded)
    # Save mask
    file.create_dataset("mask", data=mask_val_stack_expanded)

# Create a new HDF5 file
with h5py.File("data_tst.mat", "w") as file:
    # Save data_x (a_test_stack)
    file.create_dataset("data_x", data=a_test_stack_expanded)
    # Save data_y (b_test_stack)
    file.create_dataset("data_y", data=b_test_stack_expanded)
    # Save mask
    file.create_dataset("mask", data=mask_test_stack_expanded)
