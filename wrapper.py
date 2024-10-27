import torch
from torch import nn

from nmrf.config import get_cfg
from nmrf.utils import frame_utils
from nmrf.models import build_model


class Model(nn.Module):

    def __init__(self, model_name="NMRF", dataset_name="kitti"):

        super(Model, self).__init__()

        self.dataset_name = dataset_name
        cfg = self._setup_cfg(dataset_name)

        self.model = build_model(cfg)[0]
        self.model = self.model.to(torch.device("cuda"))
        checkpoint = torch.load(cfg.SOLVER.RESUME, map_location="cuda")
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict=cfg.SOLVER.STRICT_RESUME)

    def forward(self, sample):

        img1 = frame_utils.read_gen(sample[:, 0])
        img2 = frame_utils.read_gen(sample[:, 1])
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]

        sample['meta'] = []
        sample['img1'] = torch.from_numpy(img1).permute(3, 1, 2).float()
        sample['img2'] = torch.from_numpy(img2).permute(3, 1, 2).float()

        result_dict = model(sample)
        disp_pred = result_dict["disp"].detach().cpu().numpy()

        return disp_pred

    def _setup_cfg(self, dataset_name):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)

        if dataset_name == "kitti":
            config_file_path = "configs/kitti_mix_train_swint.yaml"
        elif dataset_name == "sceneflow":
            config_file_path = "configs/sceneflow_swint.yaml"
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        cfg.merge_from_file(config_file_path)
        cfg.freeze()

        return cfg
