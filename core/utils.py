import os
import argparse
from collections import defaultdict
import logging

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

from configs.default import get_cfg_defaults


def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def get_video_style(video_name, style_type):
    person_id, direction, emotion, level, *_ = video_name.split("_")
    if style_type == "id_dir_emo_level":
        style = "_".join([person_id, direction, emotion, level])
    elif style_type == "emotion":
        style = emotion
    else:
        raise ValueError("Unknown style type")

    return style


def get_style_video_lists(video_list, style_type):
    style2video_list = defaultdict(list)
    for video in video_list:
        style = get_video_style(video, style_type)
        style2video_list[style].append(video)

    return style2video_list


def get_face3d_clip(video_name, video_root_dir, num_frames, start_idx, dtype=torch.float32):
    """_summary_

    Args:
        video_name (_type_): _description_
        video_root_dir (_type_): _description_
        num_frames (_type_): _description_
        start_idx (_type_): "random" , middle, int
        dtype (_type_, optional): _description_. Defaults to torch.float32.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    video_path = os.path.join(video_root_dir, video_name)
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    length = face3d_exp.shape[0]
    clip_num_frames = num_frames
    if start_idx == "random":
        clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
    elif start_idx == "middle":
        clip_start_idx = (length - clip_num_frames + 1) // 2
    elif isinstance(start_idx, int):
        clip_start_idx = start_idx
    else:
        raise ValueError(f"Invalid start_idx {start_idx}")

    face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
    face3d_clip = torch.tensor(face3d_clip, dtype=dtype)

    return face3d_clip


def get_video_style_clip(video_path, style_max_len, start_idx="random", dtype=torch.float32):
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)

    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    return face3d_clip, pad_mask


def get_audio_name_from_video(video_name):
    audio_name = video_name[:-4] + "_seq.json"
    return audio_name


def get_audio_window(audio, win_size):
    """

    Args:
        audio (numpy.ndarray): (N,)

    Returns:
        audio_wins (numpy.ndarray): (N, W)
    """
    num_frames = len(audio)
    ph_frames = []
    for rid in range(0, num_frames):
        ph = []
        for i in range(rid - win_size, rid + win_size + 1):
            if i < 0:
                ph.append(31)
            elif i >= num_frames:
                ph.append(31)
            else:
                ph.append(audio[i])

        ph_frames.append(ph)

    audio_wins = np.array(ph_frames)

    return audio_wins


def setup_config():
    parser = argparse.ArgumentParser(description="voice2pose main program")
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume_from", type=str, default=None, help="the checkpoint to resume from")
    parser.add_argument("--test_only", action="store_true", help="perform testing and evaluation only")
    parser.add_argument("--demo_input", type=str, default=None, help="path to input for demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="the checkpoint to test with")
    parser.add_argument("--tag", type=str, default="", help="tag for the experiment")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args, cfg


def setup_logger(base_path, exp_name):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-0.5s] %(message)s")

    log_path = "{0}/{1}.log".format(base_path, exp_name)
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.handlers[0].setLevel(logging.ERROR)

    logging.info("log path: %s" % log_path)


def get_pose_params(mat_path):
    """Get pose parameters from mat file

    Args:
        mat_path (str): path of mat file

    Returns:
        pose_params (numpy.ndarray): shape (L_video, 9), angle, translation, crop paramters
    """
    mat_dict = loadmat(mat_path)

    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]

    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]

    pose_params = np.concatenate((angles, translations, crop), axis=1)

    return pose_params


def obtain_seq_index(index, num_frames, radius):
    seq = list(range(index - radius, index + radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq
