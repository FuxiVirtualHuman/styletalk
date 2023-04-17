import argparse
import cv2
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from core.networks.styletalk import StyleTalk
from core.utils import get_audio_window, get_pose_params, get_video_style_clip, obtain_seq_index
from configs.default import get_cfg_defaults


@torch.no_grad()
def get_eval_model(cfg):
    model = StyleTalk(cfg).cuda()
    content_encoder = model.content_encoder
    style_encoder = model.style_encoder
    decoder = model.decoder
    checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT)
    model_state_dict = checkpoint["model_state_dict"]
    content_encoder_dict = {k[16:]: v for k, v in model_state_dict.items() if k[:16] == "content_encoder."}
    content_encoder.load_state_dict(content_encoder_dict, strict=True)
    style_encoder_dict = {k[14:]: v for k, v in model_state_dict.items() if k[:14] == "style_encoder."}
    style_encoder.load_state_dict(style_encoder_dict, strict=True)
    decoder_dict = {k[8:]: v for k, v in model_state_dict.items() if k[:8] == "decoder."}
    decoder.load_state_dict(decoder_dict, strict=True)
    model.eval()
    return content_encoder, style_encoder, decoder


@torch.no_grad()
def render_video(
    net_G, src_img_path, exp_path, wav_path, output_path, silent=False, semantic_radius=13, fps=30, split_size=64
):

    target_exp_seq = np.load(exp_path)

    frame = cv2.imread(src_img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src_img_raw = Image.fromarray(frame)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    src_img = image_transform(src_img_raw)

    target_win_exps = []
    for frame_idx in range(len(target_exp_seq)):
        win_indices = obtain_seq_index(frame_idx, target_exp_seq.shape[0], semantic_radius)
        win_exp = torch.tensor(target_exp_seq[win_indices]).permute(1, 0)
        # (73, 27)
        target_win_exps.append(win_exp)

    target_exp_concat = torch.stack(target_win_exps, dim=0)
    target_splited_exps = torch.split(target_exp_concat, split_size, dim=0)
    output_imgs = []
    for win_exp in target_splited_exps:
        win_exp = win_exp.cuda()
        cur_src_img = src_img.expand(win_exp.shape[0], -1, -1, -1).cuda()
        output_dict = net_G(cur_src_img, win_exp)
        output_imgs.append(output_dict["fake_image"].cpu().clamp_(-1, 1))

    output_imgs = torch.cat(output_imgs, 0)
    transformed_imgs = ((output_imgs + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)

    if silent:
        torchvision.io.write_video(output_path, transformed_imgs.cpu(), fps)
    else:
        silent_video_path = "silent.mp4"
        torchvision.io.write_video(silent_video_path, transformed_imgs.cpu(), fps)
        os.system(f"ffmpeg -loglevel quiet -y -i {silent_video_path} -i {wav_path} -shortest {output_path}")
        os.remove(silent_video_path)


@torch.no_grad()
def get_netG(checkpoint_path):
    from generators.face_model import FaceGenerator
    import yaml

    with open("configs/renderer_conf.yaml", "r") as f:
        renderer_config = yaml.load(f, Loader=yaml.FullLoader)

    renderer = FaceGenerator(**renderer_config).to(torch.cuda.current_device())

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    renderer.load_state_dict(checkpoint["net_G_ema"], strict=False)

    renderer.eval()

    return renderer


@torch.no_grad()
def generate_expression_params(
    cfg, audio_path, style_clip_path, pose_path, output_path, content_encoder, style_encoder, decoder
):
    with open(audio_path, "r") as f:
        audio = json.load(f)

    audio_win = get_audio_window(audio, cfg.WIN_SIZE)
    audio_win = torch.tensor(audio_win).cuda()
    content = content_encoder(audio_win.unsqueeze(0))

    style_clip, pad_mask = get_video_style_clip(style_clip_path, style_max_len=256, start_idx=0)
    style_code = style_encoder(
        style_clip.unsqueeze(0).cuda(), pad_mask.unsqueeze(0).cuda() if pad_mask is not None else None
    )

    gen_exp_stack = decoder(content, style_code)
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose_ext = pose_path[-3:]
    pose = None
    if pose_ext == "npy":
        pose = np.load(pose_path)
    elif pose_ext == "mat":
        pose = get_pose_params(pose_path)
    # (L, 9)

    selected_pose = None
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference for demo")
    parser.add_argument(
        "--styletalk_checkpoint",
        type=str,
        default="checkpoints/styletalk_checkpoint.pth",
        help="the checkpoint to test with",
    )
    parser.add_argument(
        "--renderer_checkpoint",
        type=str,
        default="checkpoints/renderer_checkpoint.pt",
        help="renderer checkpoint",
    )
    parser.add_argument("--audio_path", type=str, default="", help="path for phoneme")
    parser.add_argument("--style_clip_path", type=str, default="", help="path for style_clip_mat")
    parser.add_argument("--pose_path", type=str, default="", help="path for pose")
    parser.add_argument("--src_img_path", type=str, default="test_images/KristiNoem1_0.jpg")
    parser.add_argument("--wav_path", type=str, default="demo/data/KristiNoem_front_neutral_level1_002.wav")
    parser.add_argument("--output_path", type=str, default="demo_output.npy", help="path for output")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.INFERENCE.CHECKPOINT = args.styletalk_checkpoint
    cfg.freeze()
    print(f"checkpoint: {cfg.INFERENCE.CHECKPOINT}")

    # load checkpoint
    with torch.no_grad():
        content_encoder, style_encoder, decoder = get_eval_model(cfg)
        exp_param_path = f"{args.output_path[:-4]}.npy"
        generate_expression_params(
            cfg,
            args.audio_path,
            args.style_clip_path,
            args.pose_path,
            exp_param_path,
            content_encoder,
            style_encoder,
            decoder,
        )

        image_renderer = get_netG(args.renderer_checkpoint)
        render_video(
            image_renderer,
            args.src_img_path,
            exp_param_path,
            args.wav_path,
            args.output_path,
            split_size=4,
        )
