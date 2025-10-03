import genesis as gs
from utils import (np2torch_clone, get_general_logger, save_curve_line, draw_curve_line)
from gsge_entity import GaussianGsPairEntities, init_log
from moviepy import VideoClip

def exp0():
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    gs.init(backend=gs.cuda)
    log = init_log(__name__, logging_level="INFO")

    raw_target_img = Image.open('../masked_rgb_frames/50.png').convert('RGB')
    target_img = np.array(raw_target_img)
    target_img = target_img / 255.0
    target_img = np2torch_clone(target_img)

    ge_entity = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
    gs_ply_root_path = '../franka_emika_panda_3dgs'

    gge = GaussianGsPairEntities(ge_entity, gs_ply_root_path,
        camera_config = {
            'res': (640, 480),
            'pos': (2.0, 2.0, 2.5),
            'lookat': (0, 0, 0.5),
            'fov_in_deg': 30,
        })
    # input("before fitting interrupt")
    info = gge.self_fit_n_fit_image(target_img, self_fit_iteration=40, iteration=2000)
    # print(info)

    save_curve_line(info['loss'], "loss", save_path="output_exp0/loss.png")

    frame_length = len(info['tdgs_img'])
    play_length = 10 # 5 seconds

    VideoClip(lambda t: info['tdgs_img'][int(t / play_length * frame_length)], duration=play_length).write_videofile('output_exp0/tdgs_frames.mp4', fps=50)

    VideoClip(lambda t: info['ge_img'][int(t / play_length * frame_length)], duration=play_length).write_videofile('output_exp0/ge_frames.mp4', fps=50)

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
            np.concatenate([np.array(loss_img), np.array(target_img)], axis=0), 
            np.concatenate([np.array(ge_img), np.array(tdgs_img)], axis=0)
        ], axis=1)
        return combined_img.astype(np.uint8)

    VideoClip(combine_frame_func, duration=play_length).write_videofile('output_exp0/combine_frames.mp4', fps=50)


if __name__ == "__main__":
    exp0()