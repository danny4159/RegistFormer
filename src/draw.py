import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import nibabel as nib
from monai.visualize import blend_images
import math
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator
import os


blend_and_transpose = lambda x, y, alpha=0.3: np.transpose(
    blend_images(x[None], y[None], alpha, cmap="hot"), (1, 2, 0)
)
"""
This lambda function blends two images and transposes the resulting image.

Parameters:
-----------
x : ndarray
    First image to blend. Should be a 2D ndarray.
y : ndarray
    Second image to blend. Should be a 2D ndarray.
alpha : float, optional
    The weight for blending the images. The higher the alpha, the more weight for the second image. Default is 0.3.

Returns:
--------
ndarray
    The blended and transposed image. Should be a 2D ndarray.

Examples:
---------
>>> img1 = np.random.rand(10, 10)
>>> img2 = np.random.rand(10, 10)
>>> blended_img = blend_and_transpose(img1, img2)
"""


def tensor2im(image_tensor, imtype=np.float32):
    # modified RKH
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)


from matplotlib.ticker import MaxNLocator
import os


def plot_images(
    images,
    labels,
    siz=4,
    vmin=0,
    vmax=2,
    cmap=None,
    colorbar=False,
    save=False,
    filename="figure.png",
):
    """
    This function plots a list of images with corresponding labels.

    Parameters:
    -----------
    images : list of ndarray
        List of images. Each image should be a 2D or 3D ndarray.
    labels : list of str
        List of labels. Each label corresponds to an image.
    siz : int, optional
        Size of each image when plotted. Default is 4.
    cmap : str, optional
        Colormap to use for displaying images. If 'gray', the image will be displayed in grayscale.
        Default is None, in which case the default colormap is used.
    save : bool, optional
        If True, save the figure to a file. Default is False.
    filename : str, optional
        Name of the file to save the figure to. Default is 'figure.png'.

    Raises:
    -------
    AssertionError
        If the number of images does not match the number of labels.

    Examples:
    ---------
    >>> img1 = np.random.rand(10, 10)
    >>> img2 = np.random.rand(10, 10)
    >>> plot_images([img1, img2], ['Image 1', 'Image 2'], save=True, filename='my_figure.png')
    """
    assert len(images) == len(labels), "Mismatch in number of images and labels"
    n = len(images)

    fig, ax = plt.subplots(ncols=n, figsize=(siz * n, siz))
    plt.subplots_adjust(wspace=0.1)  # adjust horizontal space between subplots

    for i in range(n):
        img = ax[i].imshow(images[i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax[i].axis("off")
        ax[i].set_title(labels[i], fontsize=(16 / 4) * siz)  # adjustable font size

    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(
            labelsize=(14 / 4) * siz
        )  # Increase colorbar label size (adjustable to size)
        cbar.locator = MaxNLocator(nbins=3)  # Set maximum number of tick locations
        cbar.update_ticks()

    if save:
        plt.savefig(os.path.join(os.getcwd(), filename), bbox_inches="tight", dpi=300)
    else:
        plt.show()

    return fig


def slice_array(array, block_size):
    """
    Slice a 3D NumPy array into smaller blocks along the third dimension.

    Args:
        array (numpy.ndarray): Input array to be sliced. It should have 3 dimensions.
        block_size (int): Size of each block along the third dimension.

    Returns:
        numpy.ndarray: Sliced array with shape (array.shape[0], array.shape[1], block_size, num_slices),
            where num_slices is array.shape[2] divided by block_size.

    Raises:
        AssertionError: If the input array does not have 3 dimensions or the block size does not evenly divide
            the shape of the input array along the third dimension.

    """
    assert len(array.shape) == 3, "Input array should have 3 dimensions."
    assert (
        array.shape[2] % block_size == 0
    ), "Block size should evenly divide the array shape."

    num_slices = array.shape[2] // block_size
    sliced_array = np.zeros((array.shape[0], array.shape[1], block_size, num_slices))

    for i in range(num_slices):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        sliced_array[:, :, :, i] = array[:, :, start_idx:end_idx]

    return sliced_array


def save_slices_to_nii(sliced_array, output_prefix):
    """
    Save the individual slices of a 4D NumPy array as NIfTI files.

    Args:
        sliced_array (numpy.ndarray): Input array containing the slices to be saved. It should have 4 dimensions.
        output_prefix (str): Prefix to be used for the output file names.

    Raises:
        AssertionError: If the input array does not have 4 dimensions.

    """
    assert len(sliced_array.shape) == 4, "Input array should have 4 dimensions."

    for i in range(sliced_array.shape[3]):
        data = sliced_array[:, :, :, i]
        nifti_img = nib.Nifti1Image(data, affine=np.eye(4))
        output_filename = f"{output_prefix}_{i+1}.nii.gz"
        nib.save(nifti_img, output_filename)


def calculate_mutual_info(image1, image2):
    """
    This function calculates the mutual information between two images.

    Parameters:
    image1 (np.array): The first image
    image2 (np.array): The second image

    Returns:
    float: The mutual information score
    """
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=20)
    return mutual_info_score(None, None, contingency=hist_2d)


def plot_blended_images(t1, t2, ncols=None):
    # Assuming blend_images is a function you've already defined
    # Blend images
    blended_images = np.zeros((t1.shape[0], t1.shape[2], t1.shape[3], 3))

    for i in range(t1.shape[0]):
        blended_images[i] = np.transpose(
            blend_images(t1[i], t2[i], alpha=0.15), (1, 2, 0)
        )

    # Convert the 4D array into list of 3D arrays
    list_of_arrays = [blended_images[i] for i in range(blended_images.shape[0])]

    if ncols is None:
        ncols = math.ceil(np.sqrt(len(list_of_arrays)))

    # Calculate nrows and ncols
    nrows = math.ceil(len(list_of_arrays) / ncols)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))

    # In case there are less images than slots in the grid, remove empty slots
    if nrows * ncols > len(list_of_arrays):
        for idx in range(len(list_of_arrays), nrows * ncols):
            fig.delaxes(axes.flatten()[idx])

    for idx, image in enumerate(list_of_arrays):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].imshow(image)
        axes[row, col].axis("off")  # to remove the axis

    plt.tight_layout()
    plt.show()

    return fig


def save_images(images, image_names, save_dir, epoch):
    """
    Save a list of images as PNG files.

    Args:
        images (list): A list of NumPy ndarrays representing the images to be saved.
        image_names (list): A list of strings representing the names to be used when saving the images.
        save_dir (str): The path to the directory where the images will be saved.
        epoch (int): The current epoch number to be included in the file name.

    """
    for image, name in zip(images, image_names):
        # Save the image as a PNG file
        plt.imsave(f"{save_dir}/epoch_{epoch}_{name}.png", image)


def get_current_visuals_for_pgan(real_A, fake_B, real_B):
    """
    This function prepares the visuals for a PGAN model.

    It converts the real and fake images from tensors to images and returns them in an OrderedDict.

    Args:
    real_A (torch.Tensor): Real image from the domain A.
    fake_B (torch.Tensor): Generated fake image transformed from the domain A to B.
    real_B (torch.Tensor): Real image from the domain B.

    Returns:
    OrderedDict: A dictionary containing the images. The keys are 'real_A', 'fake_B' and 'real_B' and the values are the corresponding images.
    """
    real_A = tensor2im(real_A.data)
    fake_B = tensor2im(fake_B.data)
    real_B = tensor2im(real_B.data)
    return OrderedDict([("real_A", real_A), ("fake_B", fake_B), ("real_B", real_B)])


def get_current_visuals_for_cgan(real_A, fake_A, real_B, fake_B):
    """
    This function prepares the visuals for a CGAN model.

    It converts the real and fake images from tensors to images and returns them in an OrderedDict.

    Args:
    real_A (torch.Tensor): Real image from the domain A.
    fake_A (torch.Tensor): Generated fake image similar to real image in domain A.
    real_B (torch.Tensor): Real image from the domain B.
    fake_B (torch.Tensor): Generated fake image similar to real image in domain B.

    Returns:
    OrderedDict: A dictionary containing the images. The keys are 'real_A', 'fake_A', 'real_B' and 'fake_B' and the values are the corresponding images.
    """
    real_A = tensor2im(real_A.data)
    fake_A = tensor2im(fake_A.data)
    real_B = tensor2im(real_B.data)
    fake_B = tensor2im(fake_B.data)
    return OrderedDict(
        [("real_A", real_A), ("fake_A", fake_A), ("real_B", real_B), ("fake_B", fake_B)]
    )


def plot_2d_slice(images, image_names, slice_idx):
    """
    Visualizes the 2D slices from the given 3D images at the specified slice index.

    Args:
        images (list[torch.Tensor]): List of 3D image tensors (batch_size, channel, width, height, depth) to be displayed.
        image_names (list[str]): List of names for the images for displaying as titles.
        slice_idx (int): The slice index along the depth axis to visualize the 2D slice from the 3D images.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))

    for idx, (image, title) in enumerate(zip(images, image_names)):
        slice_image = image[0, 0, :, :, slice_idx].cpu().numpy()
        axes[idx].imshow(slice_image, cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(title)

    plt.show()


def overlay_images(
    img0, img1, title0="", title_mid="", title1="", fname=None, **fig_kwargs
):
    r"""Plot two images one on top of the other using red and green channels.

    Creates a figure containing three images: the first image to the left
    plotted on the red channel of a color image, the second to the right
    plotted on the green channel of a color image and the two given images on
    top of each other using the red channel for the first image and the green
    channel for the second one. It is assumed that both images have the same
    shape. The intended use of this function is to visually assess the quality
    of a registration result.

    Parameters
    ----------
    img0 : array, shape(R, C)
        the image to be plotted on the red channel, to the left of the figure
    img1 : array, shape(R, C)
        the image to be plotted on the green channel, to the right of the
        figure
    title0 : string (optional)
        the title to be written on top of the image to the left. By default, no
        title is displayed.
    title_mid : string (optional)
        the title to be written on top of the middle image. By default, no
        title is displayed.
    title1 : string (optional)
        the title to be written on top of the image to the right. By default,
        no title is displayed.
    fname : string (optional)
        the file name to write the resulting figure. If None (default), the
        image is not saved.
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    """
    # Normalize the input images to [0,255]
    img0 = 255 * ((img0 - img0.min()) / (img0.max() - img0.min()))
    img1 = 255 * ((img1 - img1.min()) / (img1.max() - img1.min()))

    # Create the color images
    img0_red = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)
    img1_green = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)
    overlay = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)

    # Copy the normalized intensities into the appropriate channels of the
    # color images
    img0_red[..., 0] = img0
    img1_green[..., 1] = img1
    overlay[..., 0] = img0
    overlay[..., 1] = img1

    fig = _tile_plot([img0_red, overlay, img1_green], [title0, title_mid, title1])

    # If a file name was given, save the figure
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight", **fig_kwargs)

    return fig


def _tile_plot(imgs, titles, **kwargs):
    """
    Helper function
    """
    # Create a new figure and plot the three images
    fig, ax = plt.subplots(1, len(imgs))
    for ii, a in enumerate(ax):
        a.set_axis_off()
        a.imshow(imgs[ii], **kwargs)
        a.set_title(titles[ii])

    return fig
