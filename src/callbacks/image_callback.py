from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import resize
import torch
from typing import Any, List, Optional
from lightning.pytorch import Callback
from src import utils
import numpy as np
import nibabel as nib
import h5py
import os
from PIL import Image
# import itk


log = utils.get_pylogger(__name__)


class ImageLoggingCallback(Callback):
    def __init__(
        self,
        val_batch_idx: List[int] = [10, 20, 30, 40, 50],
        tst_batch_idx: List[int] = [7, 8, 9, 10, 11],
        center_crop: int = 256,
        every_epoch=5,
        log_test: bool = False,
        use_split_inference: bool = False,
    ):
        """_summary_

        Args:
            batch_idx (List[int], optional): _description_. Defaults to [10,20,30,40,50].
            log_test (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.val_batch_idx = val_batch_idx  # log images on the validation stage
        self.tst_batch_idx = tst_batch_idx  # log images on the testing stage

        self.every_epoch = every_epoch
        self.log_test = log_test  # log images on the testing stage as well
        self.center_crop = center_crop  # center crop the images to this size
        self.use_split_inference = use_split_inference
        # print("use_split_inference: ", use_split_inference)

    def saving_to_grid(self, res):
        # def gray2rgb(tensor):
        #     if tensor.size(0) == 1:  # Grayscale image with 1 channel
        #         tensor = tensor.repeat(3, 1, 1)  # Repeat the channel 3 times
        #     return tensor
        
        if not isinstance(res, (list, tuple)):
            print(f"Unexpected input to saving_to_grid: {type(res)}")
            print(f"Input content: {res}")
            return
        if len(res) == 6:
            self.ngrid = 6
            a, b, a2, b2, preds_a, preds_b = res
            err_a = torch.abs(a2 - preds_a)
            err_b = torch.abs(b2 - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
            resize(((a[0] + 1) / 2), self.first_image_size),
            resize(((a2[0] + 1) / 2), self.first_image_size),
            resize(((preds_a[0] + 1) / 2), self.first_image_size),
            resize(((b[0] + 1) / 2), self.first_image_size),
            resize(((b2[0] + 1) / 2), self.first_image_size),
            resize(((preds_b[0] + 1) / 2), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_a[0], self.first_image_size),
                resize(err_b[0], self.first_image_size),
            ])

        elif len(res) == 4:
            self.ngrid = 4
            a, b, preds_a, preds_b = res
            err_a = torch.abs(a - preds_a)
            err_b = torch.abs(b - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
                resize(((a[0] + 1) / 2), self.first_image_size),
                resize(((preds_a[0] + 1) / 2), self.first_image_size),
                resize(((b[0] + 1) / 2), self.first_image_size),
                resize(((preds_b[0] + 1) / 2), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_a[0], self.first_image_size),
                resize(err_b[0], self.first_image_size),
            ])
            
        elif len(res) == 3:
            self.ngrid = 3
            a, b, preds_b = res
            err_b = torch.abs(b - preds_b)

            if len(self.img_grid) == 0:
                self.first_image_size = a[0].shape[1:3]

            self.img_grid.extend([
                resize(((a[0] + 1) / 2), self.first_image_size),
                resize(((preds_b[0] + 1) / 2), self.first_image_size),
                resize(((b[0] + 1) / 2), self.first_image_size),
            ])
            self.err_grid.extend([
                resize(err_b[0], self.first_image_size),
            ])
            
    def on_validation_start(self, trainer, pl_module):
        self.img_grid = []
        self.err_grid = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        if len(batch[0].size()) == 5: # 3D Image
            self.val_batch_idx = [1, 2, 3, 4, 5]

        if (
            batch_idx in self.val_batch_idx
            and trainer.current_epoch % self.every_epoch == 0
        ):  
            
            if len(batch[0].size()) == 5: # 3D Image
                d_index = 30 
                evaluation_img, moving_img, fixed_img, warped_img = pl_module.model_step(batch, is_3d=True)
                evaluation_img = evaluation_img[:, :, :, :, d_index].squeeze(-1)
                moving_img = moving_img[:, :, :, :, d_index].squeeze(-1)
                fixed_img = fixed_img[:, :, :, :, d_index].squeeze(-1)
                warped_img = warped_img[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([evaluation_img, moving_img, fixed_img, warped_img])
            elif len(batch[0].size()) == 4:
                if self.use_split_inference:
                    half_size = batch[0].shape[2] // 2
                    first_half = [x[:, :, :half_size, :] for x in batch]
                    second_half = [x[:, :, half_size:, :] for x in batch]

                    res_first_half = pl_module.model_step(first_half)
                    res_second_half = pl_module.model_step(second_half)

                    if len(res_first_half) == 6 and len(res_second_half) == 6:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        a2 = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        b2 = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                        preds_a = torch.cat([res_first_half[4], res_second_half[4]], dim=2)
                        preds_b = torch.cat([res_first_half[5], res_second_half[5]], dim=2)
                        self.saving_to_grid([a, b, a2, b2, preds_a, preds_b])
                    elif len(res_first_half) == 4 and len(res_second_half) == 4:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        preds_a = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        preds_b = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                        self.saving_to_grid([a, b, preds_a, preds_b])
                    elif len(res_first_half) == 3 and len(res_second_half) == 3:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        preds_b = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        self.saving_to_grid([a, b, preds_b])
                else:
                    res = pl_module.model_step(batch)
                    self.saving_to_grid(res)
            

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if len(self.img_grid) > 0 and trainer.current_epoch % self.every_epoch == 0:
            log.info(f"Saving validation img_grid shape: <{len(self.img_grid)}>")

            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            err_grid = make_grid(
                self.err_grid, nrow=self.ngrid // 2, value_range=(0, 1)
            )

            # Log to TensorBoard
            trainer.logger.experiment.add_image(
                f"val/images", img_grid, trainer.current_epoch
            )
            trainer.logger.experiment.add_image(
                f"val/error", err_grid, trainer.current_epoch
            )
            self.img_grid = []
            self.err_grid = []
        else:
            log.debug(f"No images to log for validation")

    def on_test_start(self, trainer, pl_module):
        self.img_grid = []
        self.err_grid = [] 

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.log_test and batch_idx in self.tst_batch_idx
        ):  # log every indexes for slice number in test set
            
            if len(batch[0].size()) == 5: # 3D Image
                d_index = 30 
                evaluation_img, moving_img, fixed_img, warped_img = pl_module.model_step(batch, is_3d=True)
                evaluation_img = evaluation_img[:, :, :, :, d_index].squeeze(-1)
                moving_img = moving_img[:, :, :, :, d_index].squeeze(-1)
                fixed_img = fixed_img[:, :, :, :, d_index].squeeze(-1)
                warped_img = warped_img[:, :, :, :, d_index].squeeze(-1)
                self.saving_to_grid([evaluation_img, moving_img, fixed_img, warped_img])
            elif len(batch[0].size()) == 4:
                if self.use_split_inference:
                    half_size = batch[0].shape[2] // 2
                    first_half = [x[:, :, :half_size, :] for x in batch]
                    second_half = [x[:, :, half_size:, :] for x in batch]

                    res_first_half = pl_module.model_step(first_half)
                    res_second_half = pl_module.model_step(second_half)

                    if len(res_first_half) == 6 and len(res_second_half) == 6:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        a2 = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        b2 = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                        preds_a = torch.cat([res_first_half[4], res_second_half[4]], dim=2)
                        preds_b = torch.cat([res_first_half[5], res_second_half[5]], dim=2)
                        self.saving_to_grid([a, b, a2, b2, preds_a, preds_b])
                    elif len(res_first_half) == 4 and len(res_second_half) == 4:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        preds_a = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        preds_b = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                        self.saving_to_grid([a, b, preds_a, preds_b])
                    elif len(res_first_half) == 3 and len(res_second_half) == 3:
                        a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                        b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                        preds_b = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                        self.saving_to_grid([a, b, preds_b])
                    else:
                        res = None
                else:
                    res = pl_module.model_step(batch)
                    self.saving_to_grid(res)

    def on_test_end(self, trainer, pl_module):
        log.info(f"Saving test img_grid shape: <{len(self.img_grid)}>")

        # Create a grid of images
        if len(self.img_grid) > 0:
            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            # Log to TensorBoard
            trainer.logger.experiment.add_image(f"test/final_image", img_grid)
        else:
            log.warning(f"No images to log for testing")

#####################################################################################################################################

#####################################################################################################################################
class ImageSavingCallback(Callback):
    def __init__(self, 
                 center_crop: int = 256, 
                 subject_number_length: int = 3, 
                 test_file: str = None,
                 use_split_inference: bool = False,
                 flag_normalize: bool = True,
                 data_dir: str = None,
                 data_type:str = None,
                 ):
        """_summary_
        Image saving callback : Save images in nii format for each subject

        """
        super().__init__()
        self.center_crop = center_crop  # center crop the images to this size
        self.subject_number_length = subject_number_length
        self.test_file = test_file
        self.use_split_inference = use_split_inference
        self.flag_normalize = flag_normalize
        self.data_dir = data_dir
        self.data_type = data_type
        # print("test_file: ", test_file)
        # print("flag_normalize: ", self.flag_normalize)

    @staticmethod
    def change_torch_numpy(a, b, c, d, e=None, f=None):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim
        ), "Arrays a, b, c, and d must have the same number of dimensions"
        
        if e is not None or f is not None:
            assert (e is None or e.ndim == a.ndim) and (f is None or f.ndim == a.ndim), "Arrays e and f must have the same number of dimensions as a, b, c, and d if they are provided"

        if a.ndim == 4 or a.ndim == 5:
            a_np = a.cpu().detach().numpy()[0, 0]
            b_np = b.cpu().detach().numpy()[0, 0]
            c_np = c.cpu().detach().numpy()[0, 0]
            d_np = d.cpu().detach().numpy()[0, 0]
            e_np = e.cpu().detach().numpy()[0, 0] if e is not None else None
            f_np = f.cpu().detach().numpy()[0, 0] if f is not None else None
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        
        return a_np, b_np, c_np, d_np, e_np, f_np

    @staticmethod
    def change_numpy_nii(a, b, c, d, flag_normalize=True):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim == 3
        ), "All input arrays must have the same number of dimensions (3)"

        if flag_normalize:
            # scale to [0, 1] and [0, 255]
            a, b, c, d = (
                ((a + 1) / 2) * 255,
                ((b + 1) / 2) * 255,
                ((c + 1) / 2) * 255,
                ((d + 1) / 2) * 255,
            )

            # type to np.int16
            a, b, c, d = (
                a.astype(np.int16),
                b.astype(np.int16),
                c.astype(np.int16),
                d.astype(np.int16),
            )

        # transpose 1, 2 dim (for viewing on ITK-SNAP)
        # a, b, c, d = (
        #     np.transpose(a, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(b, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(c, axes=(1, 0, 2))[:, ::-1],
        #     np.transpose(d, axes=(1, 0, 2))[:, ::-1],
        # )

        a, b, c, d = (
            a[::-1, ::-1, :],
            b[::-1, ::-1, :],
            c[::-1, ::-1, :],
            d[::-1, ::-1, :] if d is not None else None,
        )

        # Create Nifti1Image for each
        a_nii, b_nii, c_nii, d_nii = (
            nib.Nifti1Image(a, np.eye(4)),
            nib.Nifti1Image(b, np.eye(4)),
            nib.Nifti1Image(c, np.eye(4)),
            nib.Nifti1Image(d, np.eye(4)),
        )

        return a_nii, b_nii, c_nii, d_nii
    
    @staticmethod
    def change_numpy_tif(a, b, a2, b2, preds_a, preds_b):
        a = ((a + 1) / 2 * 255).astype(np.uint8)
        b = ((b + 1) / 2 * 255).astype(np.uint8)
        a2 = ((a2 + 1) / 2 * 255).astype(np.uint8)
        b2 = ((b2 + 1) / 2 * 255).astype(np.uint8)
        preds_a = ((preds_a + 1) / 2 * 255).astype(np.uint8)
        preds_b = ((preds_b + 1) / 2 * 255).astype(np.uint8)
        return a, b, a2, b2, preds_a, preds_b

    @staticmethod
    def save_nii(a_nii, b_nii, c_nii, d_nii, subject_number, folder_path):
        nib.save(a_nii, os.path.join(folder_path, f"a_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"b_{subject_number}.nii.gz"))
        nib.save(c_nii, os.path.join(folder_path, f"preds_a_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(
                d_nii, os.path.join(folder_path, f"preds_b_{subject_number}.nii.gz")
            )
        return
    
    @staticmethod
    def save_nii_registration(a_nii, b_nii, c_nii, d_nii, subject_number, folder_path):
        nib.save(a_nii, os.path.join(folder_path, f"evaluation_img_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"moving_img_{subject_number}.nii.gz"))
        nib.save(c_nii, os.path.join(folder_path, f"fixed_img_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(
                d_nii, os.path.join(folder_path, f"warped_img_{subject_number}.nii.gz")
            )
        return

    @staticmethod
    def save_tif(a, b, a2, b2, preds_a, preds_b, subject_number, folder_path):
        a_tif = Image.fromarray(a)
        b_tif = Image.fromarray(b)
        a2_tif = Image.fromarray(a2)
        b2_tif = Image.fromarray(b2)
        preds_a_tif = Image.fromarray(preds_a)
        preds_b_tif = Image.fromarray(preds_b)
        
        a_tif.save(os.path.join(folder_path, f"{subject_number}_a.tif"))
        b_tif.save(os.path.join(folder_path, f"{subject_number}_b.tif"))
        a2_tif.save(os.path.join(folder_path, f"{subject_number}_a2.tif"))
        b2_tif.save(os.path.join(folder_path, f"{subject_number}_b2.tif"))
        preds_a_tif.save(os.path.join(folder_path, f"{subject_number}_preds_a.tif"))
        preds_b_tif.save(os.path.join(folder_path, f"{subject_number}_preds_b.tif"))
        return
    
    def saving_to_nii(self, a, b, preds_a, preds_b=None):
        if preds_a is None:
            preds_a = torch.zeros_like(a)
        if preds_b is None:
            preds_b = torch.zeros_like(b)
            
        a, b, preds_a, preds_b, _, _= self.change_torch_numpy(a, b, preds_a, preds_b)
        if a.ndim == 2: # 2D image
            self.img_a.append(a)
            self.img_b.append(b)
            self.img_preds_a.append(preds_a)
            if preds_b is not None:
                self.img_preds_b.append(preds_b)

            if len(self.img_a) == self.subject_slice_num[0]:
                a_nii = np.stack(self.img_a, -1)
                b_nii = np.stack(self.img_b, -1)
                preds_a_nii = np.stack(self.img_preds_a, -1)
                if preds_b is not None:
                    preds_b_nii = np.stack(self.img_preds_b, -1)
                else:
                    preds_b_nii = a_nii * 0  # Placeholder if preds_b is None

                # convert numpy to nii
                a_nii, b_nii, preds_a_nii, preds_b_nii = self.change_numpy_nii(
                    a_nii, b_nii, preds_a_nii, preds_b_nii, flag_normalize=self.flag_normalize
                )
                # save nii image to (.nii) file
                self.save_nii(
                    a_nii,
                    b_nii,
                    preds_a_nii,
                    preds_b_nii if preds_b is not None else None,
                    subject_number=self.dataset_list[0],
                    folder_path=self.save_folder_name,
                )

                # empty list
                self.img_a = []
                self.img_b = []
                self.img_preds_a = []
                if preds_b is not None:
                    self.img_preds_b = []
                self.dataset_list.pop(0)
                self.subject_slice_num.pop(0)

            if self.subject_number > self.subject_number_length:
                log.info(f"Saving test images up to {self.subject_number_length}")
                return
            
        elif a.ndim == 3: # 3D image
            a_nii, b_nii, preds_a_nii, preds_b_nii = self.change_numpy_nii(
                a, b, preds_a, preds_b if preds_b is not None else None,
                flag_normalize=self.flag_normalize
            )
            
            # save nii image to (.nii) file
            self.save_nii_registration(
                a_nii,
                b_nii,
                preds_a_nii,
                preds_b_nii if preds_b is not None else None,
                subject_number=self.dataset_list[0],
                folder_path=self.save_folder_name,
            )
            self.dataset_list.pop(0)

    def saving_to_tif(self, a, b, a2, b2, preds_a, preds_b=None):
        if preds_a is None:
            preds_a = torch.zeros_like(a)
        if preds_b is None:
            preds_b = torch.zeros_like(b)
            
        a, b, a2, b2, preds_a, preds_b = self.change_torch_numpy(a, b, a2, b2, preds_a, preds_b)
        a, b, a2, b2, preds_a, preds_b = self.change_numpy_tif(a, b, a2, b2, preds_a, preds_b)
        self.save_tif(
            a,
            b,
            a2,
            b2,
            preds_a,
            preds_b if preds_b is not None else None,
            subject_number=self.dataset_list.pop(0),  # Use the dataset name for saving
            folder_path=self.save_folder_name,
        )
        
    def on_test_start(self, trainer, pl_module):
        # make save folder
        folder_name = os.path.join(trainer.default_root_dir, "results")
        log.info(f"Saving test images to nifti files to {folder_name}")

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.save_folder_name = folder_name

        self.img_a = []
        self.img_b = []
        self.img_preds_a = []
        self.img_preds_b = []
        self.i = 0
        self.subject_slice_num = []
        self.subject_number = 1

        data_path = os.path.join(self.data_dir, "test", self.test_file)

        if self.data_type == 'nifti':
            with h5py.File(data_path, "r") as file:
                first_group = file[list(file.keys())[0]]
                self.dataset_list = [
                    key for key in first_group.keys()
                ]  
                self.subject_slice_num = [
                    first_group[key].shape[2] for key in self.dataset_list
                ]

        elif self.data_type == 'photo':
            with h5py.File(data_path, "r") as file:
                first_group = file[list(file.keys())[0]]
                self.dataset_list = [
                    key for key in first_group.keys()
                ]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.use_split_inference:
            half_size = batch[0].shape[2] // 2
            first_half = [x[:, :, :half_size, :] for x in batch]
            second_half = [x[:, :, half_size:, :] for x in batch]

            res_first_half = pl_module.model_step(first_half)
            res_second_half = pl_module.model_step(second_half)

            if len(res_first_half) == 6 and len(res_second_half) == 6:
                a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                a2 = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                b2 = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                preds_a = torch.cat([res_first_half[4], res_second_half[4]], dim=2)
                preds_b = torch.cat([res_first_half[5], res_second_half[5]], dim=2)
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, a2, b2, preds_a, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, a2, b2, preds_a, preds_b)
            elif len(res_first_half) == 4 and len(res_second_half) == 4:
                a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                preds_a = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                preds_b = torch.cat([res_first_half[3], res_second_half[3]], dim=2)
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, preds_a, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, preds_a, preds_b)
            elif len(res_first_half) == 3 and len(res_second_half) == 3:
                a = torch.cat([res_first_half[0], res_second_half[0]], dim=2)
                b = torch.cat([res_first_half[1], res_second_half[1]], dim=2)
                preds_b = torch.cat([res_first_half[2], res_second_half[2]], dim=2)
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, preds_b)
        else:
            if len(batch[0].size()) == 5:
                res = pl_module.model_step(batch, is_3d=True)
            elif len(batch[0].size()) == 4:
                res = pl_module.model_step(batch)
            
            if len(res) == 6:
                a, b, a2, b2, preds_a, preds_b = res
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, a2, b2, preds_a, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, a2, b2, preds_a, preds_b)
                
            if len(res) == 4:
                a, b, preds_a, preds_b = res
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, preds_a, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, preds_a, preds_b)

            elif len(res) == 3:
                a, b, preds_b = res
                if self.data_type == 'nifti':
                    self.saving_to_nii(a, b, preds_b)
                elif self.data_type == 'photo':
                    self.saving_to_tif(a, b, preds_b)

            else:
                log.error(f"Unexpected res length: {len(res)}. This case has not been implemented.")
                raise NotImplementedError("This function has not been implemented yet.")
            return
        