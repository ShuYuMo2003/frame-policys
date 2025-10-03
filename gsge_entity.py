import os
import math
from typing import List
from functools import reduce

import torch
import numpy as np
import genesis as gs

from PIL import Image
from tqdm import tqdm
from rich import print
from torch import nn
from gsplat import rasterization
from moviepy import VideoClip
from plyfile import PlyData

from utils import (np2torch_clone, quat2omega, get_general_logger, quat2mat, mat2quat,
                    normalize_torch, quat_mul, make_tensor_hook, save_curve_line, make_frame_func,
                    draw_curve_line)

############### logger #####################
class DummyLogger:
    def __getattribute__(self, name: str, /):
        raise RuntimeError(f"You should use `init_log()` to initialize the logger.")

log = DummyLogger()

def init_log(logger_name="gsge_entity", logging_level="DEBUG"):
    global log
    log = get_general_logger(logger_name, logging_level=logging_level)
    return log

class GaussianLink:
    """
        GaussianLink contains a set of Gaussians from same link.
        For example, a base of franka.
    """
    def __init__(self,
            means: torch.Tensor,
            rgbs: torch.Tensor,
            quats: torch.Tensor,
            opacities: torch.Tensor,
            scales: torch.Tensor,
            device: str='cuda:0'):

        self.means = np2torch_clone(means).requires_grad_(True)
        self.rgbs = np2torch_clone(rgbs).requires_grad_(True)
        self.quats = np2torch_clone(quats).requires_grad_(True)
        self.opacities = np2torch_clone(opacities).requires_grad_(True)
        self.scales = np2torch_clone(scales).requires_grad_(True)

        self.transform_pos = torch.zeros(self.means.shape[0], 3, device=device).requires_grad_(True)
        self.transform_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).requires_grad_(True)
        self.is_transformed = False

        self._to(device)
        # self.set_retain_transform_grad()

    def __len__(self) -> int:
        return len(self.means)

    def _to(self, device: str) -> 'GaussianLink':
        self.device = torch.device(device)

        self.means = self.means.to(self.device)
        self.rgbs = self.rgbs.to(self.device)
        self.quats = self.quats.to(self.device)
        self.opacities = self.opacities.to(self.device)
        self.scales = self.scales.to(self.device)
        self.transform_pos = self.transform_pos.to(self.device)
        self.transform_quat = self.transform_quat.to(self.device)
        return self

    def _clean_grad(self):
        self.means.grad = None
        self.rgbs.grad = None
        self.quats.grad = None
        self.opacities.grad = None
        self.scales.grad = None
        self.transform_pos.grad = None
        self.transform_quat.grad = None

    def __add__(self, other: 'GaussianLink') -> 'GaussianLink':
        if self.is_transformed or other.is_transformed:
            # we do not need any push-down-tranform machanism, because `__add__` is only used for initializing stage.
            raise RuntimeError("GaussianLink is already transformed")
        return GaussianLink(
            torch.cat([self.means, other.means.to(self.device)], dim=0),
            torch.cat([self.rgbs, other.rgbs.to(self.device)], dim=0),
            torch.cat([self.quats, other.quats.to(self.device)], dim=0),
            torch.cat([self.opacities, other.opacities.to(self.device)], dim=0),
            torch.cat([self.scales, other.scales.to(self.device)], dim=0),
            device=self.device)

    # No need. it's leaf tensor.
    # def set_retain_transform_grad(self):
    #     self.transform_pos.requires_grad_(True)
    #     self.transform_pos.retain_grad()
    #     self.transform_quat.requires_grad_(True)
    #     self.transform_quat.retain_grad()

    def set_transform(self, pos=None, quat=None):
        self.is_transformed = True
        if pos is None and quat is None:
            self.transform_pos = torch.zeros(self.means.shape[0], 3, device=self.device).requires_grad_(True)
            self.transform_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).requires_grad_(True)
        elif pos is not None and quat is not None:
            self.transform_pos = np2torch_clone(pos).requires_grad_(True)
            self.transform_quat = np2torch_clone(quat).requires_grad_(True)
        else:
            raise ValueError("pos and quat must be either both None or both not None")

        # self.set_retain_transform_grad()

    @classmethod
    def load_from_ply(cls, ply_path: str) -> 'GaussianLink':
        log.debug(f"Loading ply from {ply_path}")

        ply = PlyData.read(ply_path)
        elem = 'vertex' if 'vertex' in ply else 'point'
        v = ply[elem].data

        names = v.dtype.names
        # print("properties:", names)

        xyz = np.vstack([v['x'], v['y'], v['z']]).T  # [N,3]

        SH_0 = 0.28209479177387814
        f_dc = np.vstack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).T  # [N,3]
        rgbs = np.clip(f_dc * SH_0 + 0.5, 0.0, 1.0)

        # TODO: check v['opacity'] = opacity or `1.0 / (1.0 + np.exp(-logit))`. likely `1.0 / (1.0 + np.exp(-logit))`
        logit = v['opacity'].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-logit))

        # TODO: check v['scale_*'] = scales or exp(scales)
        raw_scales = np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T  # [N,3]
        scales = np.exp(raw_scales)
        quats = np.vstack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).T  # [N,4]
        quats = quats / (1e-10 + np.linalg.norm(quats, axis=-1, keepdims=True))

        ##### check f_rest is all zero
        f_rest_names = [n for n in names if n.startswith('f_rest_')]
        f_rest = None
        if f_rest_names:
            f_rest = np.vstack([v[n] for n in sorted(f_rest_names, key=lambda s:int(s.split('_')[-1]))]).T
            # assert f_rest all zero
            assert np.all(abs(f_rest) <= 1e-8), "f_rest is not all zero"

        nan_gs = np.isnan(xyz).any(axis=-1) | np.isnan(rgbs).any(axis=-1) | np.isnan(quats).any(axis=-1) | np.isnan(opacities).any(axis=-1) | np.isnan(scales).any(axis=-1)
        mask = ~nan_gs

        if np.sum(nan_gs) > 0:
            log.warning(f"load {ply_path} with {np.sum(nan_gs)} nan Gaussians. NaN rate is {np.sum(nan_gs) / len(nan_gs) * 100 : .2f}%")

        result = cls(xyz[mask], rgbs[mask], quats[mask], opacities[mask], scales[mask])

        return result


class GaussianEntity:
    """
        GaussianEntity is 3dgs Entity contains a set of GaussianLink.
        It can be treat as a Gaussian twin of Genesis Entity.
    """
    def __init__(self, entity: gs.engine.entities.rigid_entity.rigid_entity.RigidEntity, gs_ply_root_path: str, device: str='cuda:0'):
        self.entity = entity
        self.gs_ply_root_path = gs_ply_root_path
        self._init_3dgs_twin()
        self.device = device

    def _to(self, device):
        self.device = torch.device(device)
        for g_link_name, g_link in self.gs_twin.items():
            g_link._to(device)

    def _init_3dgs_twin(self):
        self.gs_twin = {}
        for link in self.entity.links:
            name = link.name
            vgeoms_file_list = [
                os.path.join(self.gs_ply_root_path, '.'.join([vgeom.metadata["mesh_path"].split('.')[0], 'ply']))
                for vgeom in link.vgeoms
            ]
            gset = reduce(lambda x, y: x + y,
                        (GaussianLink.load_from_ply(f) for f in vgeoms_file_list))
            self.gs_twin[name] = gset
            gset.set_transform(link.get_pos(), link.get_quat())

    def update_3dgs_twin(self):
        for link in self.entity.links:
            self.gs_twin[link.name].set_transform(link.get_pos(), link.get_quat())

    def render(self, w, h, fov_in_deg, pos, lookat, render_link_names: List[str]=['all'], device: str='cuda:0'):
        # background_color = torch.zeros(3, device=device)

        # 1. get camera intrinsic matrix
        fov_in_rad = fov_in_deg * math.pi / 180.0
        fy = 0.5 * h / math.tan(0.5 * fov_in_rad)
        fx = fy
        K = torch.tensor(
            [[fx,   0, (w)/2],
            [ 0,  fy,  (h)/2],
            [ 0,   0,       1]],
            device=device, dtype=torch.float32
        )

        # 2. get camera extrinsic matrix
        viewmat = self.get_viewmat(pos, lookat).to(device)

        # 3. get transformed means, quats, scales, opacities, rgbs
        transformed_means = []
        transformed_quats = []
        scales = []
        opacities = []
        rgbs = []

        log.debug("only render link names: %s", render_link_names)

        for g_link_name, g_link in self.gs_twin.items():
            if 'all' not in render_link_names and g_link_name not in render_link_names: continue
            # if "link0" not in g_link_name: continue
            g_link._to(device)

            ##  apply transform to means
            R_link = quat2mat(g_link.transform_quat)
            means_w = g_link.means @ R_link.T + g_link.transform_pos
            transformed_means.append(means_w)

            # apply transform to quats
            transform_quat_repeat = g_link.transform_quat.repeat(g_link.quats.shape[0], 1) # for debug
            quat_w = quat_mul(transform_quat_repeat, g_link.quats)
            # print(f"transform: {g_link.transform_quat}, quat: {g_link.quats}, quat_w: {quat_w}")

            #################### DEBUG NaN grad ####################
            # A = transform_quat_repeat.clone().detach().requires_grad_(True)
            # B = g_link.quats.clone().detach().requires_grad_(True)
            # C = quat_mul(A, B)
            # loss = torch.nn.functional.l1_loss(C, torch.zeros_like(C))

            # A.register_hook(make_tensor_hook(f"A_{g_link_name}"))
            # B.register_hook(make_tensor_hook(f"B_{g_link_name}"))
            # C.register_hook(make_tensor_hook(f"C_{g_link_name}"))

            # loss.backward()
            # print(f"A grad: {A.grad}, B grad: {B.grad}, C grad: {C.grad}")
            # breakpoint()
            # [Su]: NaN grad reason: some NaN values exist in original Gaussian data.
            #################### END OF DEBUG NaN grad ####################

            transformed_quats.append(quat_w)

            scales.append(g_link.scales)
            opacities.append(g_link.opacities)
            rgbs.append(g_link.rgbs)


        # 4. rasterize
        renders = rasterization(
            torch.cat(transformed_means, dim=0),
            torch.cat(transformed_quats, dim=0),
            torch.cat(scales, dim=0),
            torch.cat(opacities, dim=0),
            torch.cat(rgbs, dim=0),
            viewmat[None],
            K[None],
            w, h,
            packed=False,
        )[0]

        # 5. convert to uint8 image
        out_img = renders[0]
        uint8_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)

        return out_img, uint8_img

    @classmethod
    def get_viewmat(cls, pos, lookat):
        pos = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos, dtype=torch.float32)
        lookat = lookat if isinstance(lookat, torch.Tensor) else torch.tensor(lookat, dtype=torch.float32)

        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=pos.device)  # Z-up

        z = normalize_torch(lookat - pos)
        if torch.abs(torch.dot(z, up)) > 0.999:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=pos.device)

        x = normalize_torch(torch.linalg.cross(z, up))
        y = torch.linalg.cross(z, x)

        R = torch.stack([x, y, z], dim=0)
        t = -R @ pos

        viewmat = torch.eye(4, dtype=torch.float32, device=pos.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        return viewmat

    def __getattribute__(self, name: str, /):
        if name == 'joint_positions':
            return self.entity.get_dofs_position()
        else:
            return super().__getattribute__(name)


class GaussianGsPairEntities:
    """
        GaussianGsPairEntities is a pair of (Genesis Entity, `3DGS` twin(in `GaussianEntity`).
    """
    def __init__(self, ge_entity, gs_ply_root_path, camera_config={}, jnt_name_in_ge: List[str]=None, device: str='cuda:0'):
        """
            ge_entity: Genesis Entity, e.g.: create by `gs.morphs.MJCF` or `gs.morphs.URDF`
            gs_ply_root_path: root path of 3DGS ply files
            camera_config: config of camera, e.g.: `{'res': (640, 480), 'pos': (2.0, 2.0, 2.5), 'lookat': (0, 0, 0.5), 'fov_in_deg': 30}`
            jnt_name_in_ge: joint names in Genesis Entity, e.g.: `["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"]`
            device: device to use, only for `GaussianEntity` and will NOT have effect on `Genesis Entity`
        """
        self.device = device
        self._ge_scene = gs.Scene(
            show_viewer = False,
            vis_options = gs.options.VisOptions(
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
                segmentation_level = 'entity',
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
            renderer=gs.renderers.Rasterizer(),
        )
        self._ge_plane  = self._ge_scene.add_entity(gs.morphs.Plane())
        self.ge_entity = self._ge_scene.add_entity(ge_entity)

        self.camera_config = {
            'res': (640, 480),
            'pos': (2.0, 2.0, 2.5),
            'lookat': (0, 0, 0.5),
            'fov_in_deg': 30,
        }
        for key in camera_config.keys():
            self.camera_config[key] = camera_config.get(key, self.camera_config[key])

        self._ge_camera = self._ge_scene.add_camera(
            res=self.camera_config['res'],
            pos=self.camera_config['pos'],
            lookat=self.camera_config['lookat'],
            fov=self.camera_config['fov_in_deg'],
        )
        self._ge_scene.build()

        self.tdgs_entity = GaussianEntity(self.ge_entity, gs_ply_root_path, device=device)
        # sync: ge_entity -> tdgs_entity
        self.tdgs_entity.update_3dgs_twin()

        if jnt_name_in_ge is None:
            log.warning("jnt_name_in_ge is not set, use default ~~<Franka>~~ joint names.")
            self.jnt_name_in_ge = [f'joint{i}' for i in range(1, 8)] + [f'finger_joint{i}' for i in range(1, 3)]
        else:
            self.jnt_name_in_ge = jnt_name_in_ge
        self.ge_dofs_idx = reduce(lambda x, y: x + y, [self.ge_entity.get_joint(name).dofs_idx_local for name in self.jnt_name_in_ge])
        # get initial joint positions
        self.current_joint_positions = (self.ge_entity.get_dofs_position(dofs_idx_local=self.ge_dofs_idx)
                                            .detach().clone().requires_grad_(True).to(device))

    def render_3dgs(self, device: str='cuda:0', render_link_names: List[str]=['all']):
        #  def render(self, w, h, fov_in_deg, pos, lookat, device: str='cuda:0'):
        out_img, uint8_img = self.tdgs_entity.render(
            self.camera_config['res'][0],
            self.camera_config['res'][1],
            self.camera_config['fov_in_deg'],
            self.camera_config['pos'],
            self.camera_config['lookat'],
            render_link_names=render_link_names,
            device=device,
        )
        return out_img, uint8_img

    def render_ge(self, **kwargs):
        rgb, depth, seg, normal = self._ge_camera.render(**kwargs)
        return dict(rgb=rgb, depth=depth, seg=seg, normal=normal)

    def self_fit(self, self_fit_iteration, update_parameter_names=["rgbs", "opacities"]):
        image = self.render_ge(rgb=True, segmentation=True)
        target_img, seg_img = image['rgb'], image['seg']
        masked_target_img = target_img * (seg_img != 1)[:, :, None]

        # uint8 to float32
        masked_target_img = masked_target_img.astype(np.float32) / 255.0

        info = self.fit_image(masked_target_img, dof_lr=1e-8, color_lr=1e-3, update_parameter_names=update_parameter_names, iteration=self_fit_iteration)
        return info

    def self_fit_n_fit_image(self, image, dof_lr=0.01, color_lr=1e-5, update_parameter_names: List[str] | None = None, self_fit_iteration=1000, iteration=1000):
        log.info("start self fitting.")
        info_0 = self.self_fit(self_fit_iteration=self_fit_iteration)
        log.info("self fitting done.")

        log.info("start fitting image.")
        info_1 = self.fit_image(image, dof_lr, color_lr, update_parameter_names, iteration)
        log.info("fitting image done.")

        info = {}
        for key in info_0.keys():
            info[key] = info_0[key] + info_1[key]
        return info

    def fit_image(self, image, dof_lr=0.01, color_lr=1e-5, update_parameter_names: List[str] | None = None, iteration=1000):
        """
            will update the params in `update_parameter_names` and `joint_positions`.

            target_img: (3, H, W) in float32, numpy or torch tensor
        """
        target_img = np2torch_clone(image).detach().clone().to(self.device)
        
        numpy_target_img = image.copy() if isinstance(image, np.ndarray) else image.detach().clone().cpu().numpy()
        numpy_target_img = (numpy_target_img * 255.0).astype(np.uint8)

        #### prepare update parameters
        if update_parameter_names is None:
            update_parameter_names = ["rgbs", "opacities"]
        update_parameters = []; parameter_name_list = [] # for debug only.
        for name in update_parameter_names:
            for g_link_name, g_link in self.tdgs_entity.gs_twin.items():
                update_parameters.append(getattr(g_link, name))
                parameter_name_list.append(name + "_" + g_link_name)
        # update_parameters.append(self.current_joint_positions)
        # parameter_name_list.append("current_joint_positions")
        log.info("update parameters: %s", parameter_name_list)
        for t, name in zip(update_parameters, parameter_name_list):
            log.debug(f"name: {name}, is_leaf: {t.is_leaf}, grad_fn: {t.grad_fn}, type: {type(t)}")
        #### end of prepare update parameters

        render_link_order = list(self.tdgs_entity.gs_twin.keys())
        # log.info("render link order: %s", render_link_order)

        #### prepare optimizer
        optimizer = torch.optim.Adam([
            {"params": update_parameters, "lr": color_lr},
            {"params": self.current_joint_positions, "lr": dof_lr},
        ])
        stat_info = dict(loss=[], tdgs_img=[], ge_img=[], target_img=[], consider_links=[])
        pbar = tqdm(range(iteration), "Fitting Image.", total=iteration, )
        for iter_idx in pbar:
            # which link will be rendered and fit in this iteration
            # consider_links = render_link_order[:iter_idx // (iteration // len(render_link_order)) + 1]
            consider_links = render_link_order
            tdgs_img, tdgs_uint8_img = self.render_3dgs(render_link_names=consider_links)
            loss = torch.nn.functional.l1_loss(tdgs_img, target_img)
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()

            stat_info['loss'].append(loss.item())
            stat_info['tdgs_img'].append(tdgs_uint8_img)
            stat_info['ge_img'].append(self.render_ge()['rgb'])
            stat_info['target_img'].append(numpy_target_img)
            stat_info['consider_links'].append(consider_links)

            pbar.set_description(f"Loss: {loss.item():.5f}")

            ###### [BEGIN] compute current_joint_positions grad MANUALLY ######
            transform_grad = {}
            for ge_link in self.ge_entity.links:
                if ge_link.name not in consider_links: continue
                # log.debug(f"compute grad for ge_link: {ge_link.name}")
                tdgs_link = self.tdgs_entity.gs_twin[ge_link.name]
                name = ge_link.name
                transform_grad[name] = {}

                J = self.ge_entity.get_jacobian(ge_link)
                J_v, J_w = J[0:3, :], J[3:6, :]

                dpos_dq = J_v # d pos / d joint_positions (3, n_dofs)

                domega_dq = J_w # d omega / d joint_positions (3, n_dofs)
                current_link_quat = normalize_torch(tdgs_link.transform_quat)
                current_link_omega = quat2omega(current_link_quat)
                # TODO: check if this is correct
                dquat_dq = 0.5 * (current_link_omega @ domega_dq)

                transform_grad[name]['dpos_dq'] = dpos_dq
                transform_grad[name]['dquat_dq'] = dquat_dq

                dloss_dpos = tdgs_link.transform_pos.grad
                dloss_dquat = tdgs_link.transform_quat.grad

                # TODO: check if this is correct
                dloss_dq = dloss_dpos @ dpos_dq + dloss_dquat @ dquat_dq

                transform_grad[name]['dloss_dq'] = dloss_dq

            dloss_dq_list = [transform_grad[name]['dloss_dq'] for name in transform_grad.keys()]
            dloss_dq_mean = torch.stack(dloss_dq_list, dim=0).mean(dim=0)

            self.current_joint_positions.grad = dloss_dq_mean
            ###### [END] compute current_joint_positions grad MANUALLY ######

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # have to clean grad explicitly, some leaf tensor like `transform_pos` and `transform_quat` will not be cleaned by `optimizer.zero_grad(set_to_none=True)`
            for g_link_name, g_link in self.tdgs_entity.gs_twin.items():
                g_link._clean_grad()

            # use new joint positions to update `ge_entity` first and then sync to `tdgs_entity`
            # this step will trigger pytorch warning:
            ### /mnt/afs/sujiayi/miniconda3/envs/vipolicy/lib/python3.12/site-packages/torch/utils/_device.py:104: UserWarning:
            ### The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated
            ### during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad()
            ### on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.
            ### See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at /pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
            # TODO: this is diect call of `Genesis` API. check if have bad effect. ignore it for now.
            self.ge_entity.set_dofs_position(self.current_joint_positions.detach().clone())
            self._ge_scene.step()

            self.tdgs_entity.update_3dgs_twin()

        return stat_info

