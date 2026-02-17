import logging
import os
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetPathLoss(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, split: str,
        path_response_ghz: Literal['2.6GHz', '6GHz', '28GHz', '60GHz', '100GHz'],
        min_path_loss: float, max_path_loss: float, img_size: int,
        eps_image: int, output_kernel_size: int, *args, **kwargs
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__scenario_path: str = os.path.join(data_path, scenario)
        self._split: str = split
        self._path_response_ghz = path_response_ghz
        self.__output_kernel_size: int = output_kernel_size
        self._img_size = img_size
        self._eps_image = eps_image
        self._min_path_loss = min_path_loss
        self._max_path_loss = max_path_loss
        
        self.__num_bss_per_env = 5
        self._num_ues_per_env = 30
        self._num_pairs_per_env = self.__num_bss_per_env * self._num_ues_per_env
        
        self._environments: list[str] = self.__prepare_environments()
    
    def __getitem__(self, pair_idx: int):
        environment_idx: int = pair_idx // self._num_pairs_per_env
        
        local_pair_idx = pair_idx % self._num_pairs_per_env
        ue_idx = local_pair_idx % self._num_ues_per_env
        
        environment: str = self._environments[environment_idx]
        
        env_path = os.path.join(self.__scenario_path, environment)
        metadata = dict(np.load(os.path.join(env_path, "metadata.npz")))
        
        bs_path_losses = []
        for bs_idx in range(self.__num_bss_per_env):
            pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
            pair_data_path = os.path.join(pair_path, "data.npz")
            pair_data = dict(np.load(pair_data_path, allow_pickle=True))
            path_response = pair_data["path_responses"][()][self._path_response_ghz]
            magnitude = np.sum(np.abs(path_response))
            path_loss = -10 * np.log10(magnitude ** 2)
            path_loss = (path_loss - self._min_path_loss) / (self._max_path_loss - self._min_path_loss)
            bs_location = pair_data["locations"][()]["bs"][:2][::-1] * self._img_size
            bs_path_losses.append((path_loss, bs_location))
        
        input_img = list(np.load(os.path.join(pair_path, "input_img.npz")).values())[0].astype(np.float32)[0]
        assert input_img.shape[-1] == self._img_size
        
        map_img = -1 * np.expand_dims(input_img, axis=0).repeat(3, axis=0)
        map_img[2] = np.zeros_like(map_img[2])
        
        for path_loss, bs_loc_y_x in bs_path_losses:
            map_img[0][
            max(0, int(bs_loc_y_x[0]) - self._eps_image): int(bs_loc_y_x[0]) + self._eps_image,
            max(0, int(bs_loc_y_x[1]) - self._eps_image): int(bs_loc_y_x[1]) + self._eps_image,
            ] = bs_loc_y_x[0] / self._img_size
            
            map_img[1][
            max(0, int(bs_loc_y_x[0]) - self._eps_image): int(bs_loc_y_x[0]) + self._eps_image,
            max(0, int(bs_loc_y_x[1]) - self._eps_image): int(bs_loc_y_x[1]) + self._eps_image,
            ] = bs_loc_y_x[1] / self._img_size
            
            map_img[2][
            max(0, int(bs_loc_y_x[0]) - self._eps_image): int(bs_loc_y_x[0]) + self._eps_image,
            max(0, int(bs_loc_y_x[1]) - self._eps_image): int(bs_loc_y_x[1]) + self._eps_image,
            ] = path_loss
        
        bs_path_losses = np.vstack(
            [np.concatenate(([path_loss], loc / self._img_size)) for path_loss, loc in bs_path_losses]
        )
        
        locations = pair_data["locations"].item()
        ue_location = locations['ue'].astype(np.float32)[:2].astype(np.float32) * max(input_img.shape)
        ue_location = ue_location[::-1]
        
        ue_loc_img = list(np.load(os.path.join(pair_path, "ue_loc_img.npz")).values())[0].astype(np.float32)
        ue_loc_img = resize(ue_loc_img, (1, 224, 224))
        output_kernel_size = self.__output_kernel_size
        ue_loc_img = gaussian_filter(ue_loc_img, output_kernel_size)
        ue_loc_img_max = ue_loc_img.max()
        if ue_loc_img_max != 0:
            ue_loc_img /= ue_loc_img_max
        
        return (
            map_img,
            bs_path_losses.astype(np.float32),
            ue_loc_img,
            metadata["img_size"] / 2,
            ue_location,
        )
    
    def __len__(self) -> int:
        # return 88
        return len(self._environments) * self._num_pairs_per_env
    
    def __prepare_environments(self) -> list[str]:
        environments = os.listdir(self.__scenario_path)
        environments = sorted(filter(str.isnumeric, environments))
        
        if self._split == "train":
            return sorted(set(environments[:900] + environments[1000: 9499]))
        elif self._split == "val":
            return sorted(set(environments[9499:]))
        else:
            return environments[900:1000]
