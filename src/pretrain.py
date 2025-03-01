# Perform hard-coding for now

import matplotlib
import argparse

matplotlib.use("Agg")

import torch
import time
import random
import os
import json

from tqdm import tqdm

from matplotlib import pyplot as plt

from src.models.components.autoencoder_kl import Encoder, AutoencoderKL
from src.models.components.perceptual import PerceptualLoss

from src.data.components.transforms import dataset_IXI
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F


def imshow_latent_images(images, out_path):
    """Shows latent images with multiple channels as separate grayscale images.

    Args:
        images: numpy array of shape (1, C, H, W) where C is the number of channels, H is height, and W is width.
    """
    assert len(images.shape) == 4, "Input images should have 4 dimensions (1, C, H, W)"
    assert images.shape[0] == 1, "This function supports single image only"

    images = images.detach().cpu().numpy()
    num_channels = images.shape[1]

    if num_channels > 5:
        # perform PCA
        pca = PCA(n_components=5)
        _, C, H, W = images.shape
        reshaped_image = images.reshape(C, H * W)
        pca_result = pca.fit_transform(reshaped_image.T)
        pca_images = pca_result.T.reshape((-1, H, W))
        # Normalize each component to [0, 1]
        pca_images = (pca_images - pca_images.min(axis=(1, 2), keepdims=True)) / (
            pca_images.max(axis=(1, 2), keepdims=True)
            - pca_images.min(axis=(1, 2), keepdims=True)
        )
        images = pca_images[None]
    num_channels = images.shape[1]
    # Create a figure with sub-plots
    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 5, 5))

    # Show each channel as a grayscale image
    for i in range(num_channels):
        channel_image = images[0, i, :, :]
        axes[i].imshow(channel_image, cmap="gray")
        axes[i].set_title(f"Channel {i+1}")
        axes[i].axis("off")

    # Show the figure
    plt.savefig(out_path)
    plt.close()
    return


parser = argparse.ArgumentParser(description="Pretraining script")

# Add the arguments
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/kanghyun/misalign-benchmark/data/IXI/train/prepared_data_0_0.h5",
    help="Path to the data directory",
)
parser.add_argument(
    "--val_dir",
    type=str,
    default="/home/kanghyun/misalign-benchmark/data/IXI/val/prepared_data_0.0_0.0.h5",
    help="Path to the data directory",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/kanghyun/misalign-benchmark/logs/pretrain",
    help="Path to the output directory",
)
# Add the arguments
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--nce", type=bool, default=True, help="NCE ON/OFF")
parser.add_argument("--nce_weight", type=float, default=0.01, help="NCE weight")
parser.add_argument("--val_interval", type=int, default=10, help="Validation interval")
parser.add_argument("--batch_size", type=int, default=3, help="Batch Size")

parser.add_argument(
    "--num_res_blocks",
    type=int,
    nargs=2,
    default=[2, 2],
    help="Number of residual blocks",
)
parser.add_argument(
    "--num_channels", type=int, nargs=2, default=[32, 64], help="Number of channels"
)
parser.add_argument(
    "--latent_channels", type=int, default=3, help="Number of latent channels"
)

# Parse the arguments
args = parser.parse_args()

# Create a subdirectory name based on the values of nce and nce_weight
subdir_name = f'nce_{"on" if args.nce else "off"}_weight_{args.nce_weight}'

# Create the subdirectory in output_dir
args.output_dir = os.path.join(args.output_dir, subdir_name)
os.makedirs(args.output_dir, exist_ok=True)

# Save the arguments to a JSON file in the output directory
with open(f"{args.output_dir}/args.json", "w") as f:
    json.dump(vars(args), f, indent=4)

### Network architecture
spatial_dims = 2
in_channels = 1
out_channels = 1
num_res_blocks = args.num_res_blocks
num_channels = args.num_channels
latent_channels = args.latent_channels
n_epochs = args.n_epochs
nce_weight = args.nce_weight
nce = args.nce
output_dir = args.output_dir
### network params
perceptual_weight = 0.05

kl_weight = 1e-6

val_interval = 10
epoch_recon_loss_list = []
epoch_gen_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4
#####################

train_dataset = dataset_IXI(
    data_dir=args.data_dir,
    flip_prob=0.5,
    rot_prob=0.5,
    rand_crop=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=3,
    pin_memory=False,
    shuffle=True,
)

val_dataset = torch.utils.data.Subset(
    dataset_IXI(
        data_dir=args.val_dir,
        flip_prob=0.0,
        rot_prob=0.0,
    ),
    (range(0, 50)),
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=1, num_workers=3, pin_memory=False, shuffle=False
)

print(len(train_dataset), len(val_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model = AutoencoderKL(
    in_channels=in_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    num_channels=num_channels,
    latent_channels=latent_channels,
)
model.to(device)

perceptual_loss = PerceptualLoss(
    spatial_dims=spatial_dims, network_type="radimagenet_resnet50"
)
perceptual_loss.to(device)

optimizer_g = torch.optim.Adam(params=model.parameters(), lr=1e-4)
l1_loss = torch.nn.L1Loss()


def random_rotate_flip_scale(image):
    rotations = [
        0,
        1,
        2,
        3,
    ]  # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
    num_rotations = random.choice(rotations)

    # Flip along x, y or both randomly
    flips = [
        None,
        [-1],
        [-2],
        [-1, -2],
    ]  # None: no flip, -1: flip along x, -2: flip along y, [-1, -2]: flip along both x and y
    flip_dims = random.choice(flips)
    # scale = random.uniform(0.8, 1.2)

    scale = 1.00  # no scaling, output is normalized

    rotated_image = torch.rot90(image, num_rotations, [-1, -2])
    if flip_dims is not None:
        rotated_image = torch.flip(rotated_image, flip_dims)
    rotated_image = rotated_image * scale

    return (
        rotated_image,
        -num_rotations,
        flip_dims,
        scale,
    )  # Also return the flip dimensions for reversing


def random_rotate_flip_scale_inverse(image, rotations, flip_dims, scale):
    # Calculate the correct number of rotations to reverse
    reverse_rotated_image = image / scale

    if flip_dims is not None:
        reverse_rotated_image = torch.flip(reverse_rotated_image, flip_dims)

    reverse_rotations = rotations
    x = torch.rot90(reverse_rotated_image, reverse_rotations, [-1, -2])
    return x


def batch_random_rotate_flip_scale(images):
    batch_size = images.shape[0]
    transformed_images = []
    rotations = []
    flip_dims = []
    scales = []

    for i in range(batch_size):
        transformed_image, rotation, flip_dim, scale = random_rotate_flip_scale(
            images[i]
        )
        transformed_images.append(transformed_image)
        rotations.append(rotation)
        flip_dims.append(flip_dim)
        scales.append(scale)

    return torch.stack(transformed_images), rotations, flip_dims, scales


def batch_random_rotate_flip_scale_inverse(images, rotations, flip_dims, scales):
    batch_size = images.shape[0]
    reversed_images = []

    for i in range(batch_size):
        reversed_image = random_rotate_flip_scale_inverse(
            images[i], rotations[i], flip_dims[i], scales[i]
        )
        reversed_images.append(reversed_image)

    return torch.stack(reversed_images)


# Modifying the function to use fixed transformations


def fixed_rotate_flip_scale(image):
    num_rotations = 1
    flip_dims = [-1]
    scale = 1.2

    rotated_image = torch.rot90(image, num_rotations, [-1, -2])
    if flip_dims is not None:
        rotated_image = torch.flip(rotated_image, flip_dims)
    rotated_image = rotated_image * scale

    return rotated_image, -num_rotations, flip_dims, scale


#### Test the functions ####
original_image = torch.rand(1, 64, 64)
# Apply fixed transformations
transformed_image, rotations, flip_dims, scale = fixed_rotate_flip_scale(original_image)
# Revert the transformations
reversed_image = random_rotate_flip_scale_inverse(
    transformed_image, rotations, flip_dims, scale
)
# Maximum absolute difference between the original and the reversed image
max_diff = torch.max(torch.abs(original_image - reversed_image))
print(max_diff)
if max_diff > 1e-6:
    raise ValueError("The functions are not working as expected")

original_image = torch.rand(4, 1, 64, 64)
# Apply fixed transformations
transformed_image, rotations, flip_dims, scale = batch_random_rotate_flip_scale(
    original_image
)
# Revert the transformations
reversed_image = batch_random_rotate_flip_scale_inverse(
    transformed_image, rotations, flip_dims, scale
)
# Maximum absolute difference between the original and the reversed image
max_diff = torch.max(torch.abs(original_image - reversed_image))
print(max_diff)
if max_diff > 1e-6:
    raise ValueError("The functions are not working as expected")

#############################################

cross_entropy_loss = torch.nn.CrossEntropyLoss()


def PatchNCELoss(f_q, f_k, tau=0.07):
    # batch size, channel size, and number of sample locations
    B, C, S = f_q.shape

    # calculate v * v+: BxSx1
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S, device=f_q.device, dtype=torch.bool)[None, :, :]
    l_neg.masked_fill_(identity_matrix, -10.0)

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return PatchNCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long, device=f_q.device)
    return cross_entropy_loss(predictions, targets)


total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    recon_epoch_loss = 0
    kl_epoch_loss = 0
    nce_epoch_loss = 0
    p_epoch_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        optimizer_g.zero_grad(set_to_none=True)
        x, y = batch
        images = x
        images2 = y

        if images.shape[-1] % 16 != 0 or images.shape[-2] % 16 != 0:
            new_height = images.shape[-2] - (images.shape[-2] % 16)
            new_width = images.shape[-1] - (images.shape[-1] % 16)

            images = images[..., :new_height, :new_width]
            images2 = images2[..., :new_height, :new_width]

        images = images.to(device)
        reconstruction, mu, sigma, _ = model(images)
        recons_loss1 = l1_loss(reconstruction.float(), images.float())
        kl_loss1 = 0.5 * torch.sum(
            mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1, 2, 3]
        )
        kl_loss1 = torch.sum(kl_loss1) / kl_loss1.shape[0]
        p_loss1 = perceptual_loss(reconstruction.float(), images.float())

        ## This part is for image2

        images2 = images2.to(device)
        reconstruction, mu, sigma, _ = model(images2)
        recons_loss2 = l1_loss(reconstruction.float(), images2.float())
        kl_loss2 = 0.5 * torch.sum(
            mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=[1, 2, 3]
        )
        kl_loss2 = torch.sum(kl_loss2) / kl_loss2.shape[0]
        p_loss2 = perceptual_loss(reconstruction.float(), images2.float())

        recons_loss = (recons_loss1 + recons_loss2) / 2
        kl_loss = (kl_loss1 + kl_loss2) / 2
        p_loss = (p_loss1 + p_loss2) / 2

        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        ################## PatchNCE Loss ##################

        if nce:
            batch_size = images.shape[0]

            A1, A2 = model.NCEHead(images)
            rot_images, rotations, flip_dims, scale = batch_random_rotate_flip_scale(
                images
            )
            rot_A1, rot_A2 = model.NCEHead(rot_images)

            rot_A1 = batch_random_rotate_flip_scale_inverse(
                rot_A1, rotations, flip_dims, scale
            )

            rot_A2 = batch_random_rotate_flip_scale_inverse(
                rot_A2, rotations, flip_dims, scale
            )

            nce_loss_A = PatchNCELoss(
                A1.reshape(A1.shape[0], A1.shape[1], -1),
                rot_A1.reshape(rot_A1.shape[0], rot_A1.shape[1], -1),
                tau=0.07,
            ) + PatchNCELoss(
                A2.reshape(A2.shape[0], A2.shape[1], -1),
                rot_A2.reshape(rot_A2.shape[0], rot_A2.shape[1], -1),
                tau=0.07,
            )  # Two pairs of samples

            B1, B2 = model.NCEHead(images2)
            rot_images2, rotations, flip_dims, scale = batch_random_rotate_flip_scale(
                images2
            )
            rot_B1, rot_B2 = model.NCEHead(rot_images2)

            rot_B1 = batch_random_rotate_flip_scale_inverse(
                rot_B1, rotations, flip_dims, scale
            )

            rot_B2 = batch_random_rotate_flip_scale_inverse(
                rot_B2, rotations, flip_dims, scale
            )

            nce_loss_B = PatchNCELoss(
                B1.reshape(B1.shape[0], B1.shape[1], -1),
                rot_B1.reshape(rot_B1.shape[0], rot_B1.shape[1], -1),
                tau=0.07,
            ) + PatchNCELoss(
                B2.reshape(B2.shape[0], B2.shape[1], -1),
                rot_B2.reshape(rot_B2.shape[0], rot_B2.shape[1], -1),
                tau=0.07,
            )  # Two pairs of samples

            nce_loss = (nce_loss_A + nce_loss_B) / 2

            loss_g = loss_g + nce_weight * nce_loss
        else:
            nce_loss = loss_g * 0
        ############################################
        loss_g.backward()
        optimizer_g.step()

        recon_epoch_loss += recons_loss.item()
        kl_epoch_loss += kl_loss.item() * kl_weight
        nce_epoch_loss += nce_loss.item() * nce_weight
        p_epoch_loss += p_loss.item() * perceptual_weight

        progress_bar.set_postfix(
            {
                "recon": recon_epoch_loss / (step + 1),
                "kl": kl_epoch_loss / (step + 1),
                "nce": nce_epoch_loss / (step + 1),
                "percep": p_epoch_loss / (step + 1),
            }
        )

    epoch_recon_loss_list.append(recon_epoch_loss / (step + 1))

    if (epoch) % 1 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                x, y = batch
                reconstruction, latent_img, _, _ = model(x.to(device))
                val_loss += l1_loss(reconstruction.float(), x.to(device).float())

                if val_step == 30 and epoch % val_interval == 0:
                    plt.figure(dpi=300)
                    plt.subplot(1, latent_img.shape[1] + 2, 1)
                    plt.imshow(x[0, 0].detach().cpu(), cmap="gray")
                    plt.axis("off")

                    plt.subplot(1, latent_img.shape[1] + 2, 2)
                    plt.imshow(reconstruction[0, 0].detach().cpu(), cmap="gray")
                    plt.axis("off")

                    for _d in range(latent_img.shape[1]):
                        plt.subplot(1, latent_img.shape[1] + 2, _d + 3)
                        plt.imshow(latent_img[0, _d].detach().cpu(), cmap="gray")
                        plt.axis("off")
                    plt.title("latent image of T1")
                    #                    plt.show()
                    plt.savefig(os.path.join(output_dir, f"latent_T1_{epoch}.png"))
                    plt.close()

                reconstruction, latent_img, _, _ = model(y.to(device))
                val_loss += l1_loss(reconstruction.float(), y.to(device).float())
                if val_step == 30 and epoch % val_interval == 0:
                    plt.figure(dpi=300)
                    plt.subplot(1, latent_img.shape[1] + 2, 1)
                    plt.imshow(y[0, 0].detach().cpu(), cmap="gray")
                    plt.axis("off")

                    plt.subplot(1, latent_img.shape[1] + 2, 2)
                    plt.imshow(reconstruction[0, 0].detach().cpu(), cmap="gray")
                    plt.axis("off")
                    for _d in range(latent_img.shape[1]):
                        plt.subplot(1, latent_img.shape[1] + 2, _d + 3)
                        plt.imshow(latent_img[0, _d].detach().cpu(), cmap="gray")
                        plt.axis("off")
                    plt.title("latent image of T2")
                    #                   plt.show()

                    plt.savefig(os.path.join(output_dir, f"latent_T2_{epoch}.png"))

                    plt.close()

        val_loss /= val_step
        print(f"val_loss at epoch {epoch}: {val_loss}")
        val_recon_epoch_loss_list.append(val_loss)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# Saving model
encoder = model.encoder
torch.save(
    encoder.state_dict(),
    os.path.join(output_dir, "encoder_pretrain.pth"),
)

# Loading it to model
encoder = Encoder(
    in_channels=1,
    num_channels=num_channels,
    out_channels=num_channels[-1],
    num_res_blocks=num_res_blocks,
)  # You need to define the Encoder class somewhere
# Load the state dict
encoder.load_state_dict(torch.load(os.path.join(output_dir, "encoder_pretrain.pth")))
encoder = encoder.to(device)

im1, im2 = val_dataset[40]
im = im2[None].to(device)

with torch.no_grad():
    _, out = encoder(im)

imshow_latent_images(
    out[0],
    os.path.join(output_dir, "down_sample_latent_1.png"),
)
imshow_latent_images(
    out[1],
    os.path.join(output_dir, "down_sample_latent_2.png"),
)

print("Finished!!")
