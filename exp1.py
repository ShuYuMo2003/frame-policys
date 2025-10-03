import genesis as gs
from functools import reduce
from pathlib import Path
from tqdm import tqdm
from utils import (np2torch_clone, get_general_logger, save_curve_line, draw_curve_line)
from gsge_entity import GaussianGsPairEntities, init_log
from moviepy import VideoClip


def exp1():
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    gs.init(backend=gs.cuda)
    log = init_log(__name__, logging_level="INFO")

    #### prepare frames
    frames_list = list(map(lambda x : str(x), Path('../masked_rgb_frames').glob('*.png')))
    frames_list.sort(key=lambda x : int(x.split('/')[-1].split('.')[0]))
    frames = list(map(lambda x : np2torch_clone(np.array(Image.open(x).convert('RGB')) / 255.0), frames_list))

    ge_entity = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    gs_ply_root_path = '../franka_emika_panda_3dgs'


    #### prepare gge
    gge = GaussianGsPairEntities(ge_entity, gs_ply_root_path,
        camera_config = {
            'res': (640, 480),
            'pos': (2.0, 2.0, 2.5),
            'lookat': (0, 0, 0.5),
            'fov_in_deg': 30,
        })
    
    #### fit images.
    single_info = gge.self_fit(self_fit_iteration=100, update_parameter_names=["rgbs"])
    infos = [single_info]
    for frame in tqdm(frames[:100:2], desc="Fitting images."):
        single_info = gge.self_fit(self_fit_iteration=40, update_parameter_names=["rgbs"])
        # infos.append(single_info)

        single_info = gge.fit_image(frame, dof_lr=0.05, color_lr=1e-5, iteration=100)
        infos.append(single_info)
    
    info = {}
    for key in infos[0].keys():
        info[key] = reduce(lambda x, y : x + y, [info[key] for info in infos])


    #### save results
    log.info("Total frames: %d", len(frames))
    save_curve_line(info['loss'], "loss", save_path="output_exp1/loss.png")


    frame_length = len(info['loss'])
    play_length = 20
    VideoClip(lambda t: info['tdgs_img'][int(t / play_length * frame_length)], duration=play_length).write_videofile('output_exp1/tdgs_frames.mp4', fps=50)
    VideoClip(lambda t: info['ge_img'][int(t / play_length * frame_length)], duration=play_length).write_videofile('output_exp1/ge_frames.mp4', fps=50)

    def combine_frame_func(t):
        t = int(t / play_length * frame_length)
        loss_img = draw_curve_line(info['loss'][:t], "loss", xlabel="Iteration", ylabel="Loss")
        tdgs_img = info['tdgs_img'][t]
        ge_img = info['ge_img'][t]
        target_img = info['target_img'][t]
        # make it same size
        loss_img = Image.fromarray(loss_img).convert('RGB')
        tdgs_img = Image.fromarray(tdgs_img).convert('RGB')
        ge_img = Image.fromarray(ge_img).convert('RGB')
        target_img = Image.fromarray(target_img).convert('RGB')

        loss_img = loss_img.resize((tdgs_img.width, tdgs_img.height))
        ge_img = ge_img.resize((tdgs_img.width, tdgs_img.height))
        target_img = target_img.resize((tdgs_img.width, tdgs_img.height))

        combined_img = np.concatenate([
            np.concatenate([np.array(loss_img), np.array(target_img)], axis=1), 
            np.concatenate([np.array(ge_img), np.array(tdgs_img)], axis=1)
        ], axis=0)
        return combined_img.astype(np.uint8)

    VideoClip(combine_frame_func, duration=play_length).write_videofile('output_exp1/combine_frames.mp4', fps=50)


if __name__ == "__main__":
    exp1()