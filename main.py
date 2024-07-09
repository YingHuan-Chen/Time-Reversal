import torch

from models.pipeline_time_reversal import TimeReversalPipeline
from diffusers.utils import load_image, export_to_video

import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--outputdir", type=str, default="./outputs/room", help="output directory"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/room", help="input data directory"
    )
    parser.add_argument('--w_o_noise_re_injection', action='store_true')
    parser.add_argument("--s_churn", type=float, default=0.5, help="churn tern of scheduler")
    parser.add_argument("--t0", type=int, default=5, help="Cutoff timestep index for noise injection")
    parser.add_argument("--M", type=int, default=8, help="Number of noise injection steps")

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()

def main(args):
    pipe = TimeReversalPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Load the conditioning image
    image_1 = load_image("data/room/room_00010_00018-0000.jpg")
    image_1 = image_1.resize((1024, 576))

    image_2 = load_image("data/room/room_00010_00018-0001.jpg")
    image_2 = image_2.resize((1024, 576))

    generator = torch.manual_seed(42)

    if args.w_o_noise_re_injection:
        t0 = 0
    else:
        t0 = args.t0
    
    frames = pipe(image_1, image_2, s_churn=args.s_churn, M=args.M, t0=t0, decode_chunk_size=8, generator=generator).frames[0] 
    export_to_video(frames, f"outputs/room/generated.mp4", fps=7)


if __name__ == "__main__":
    args = config_parser()
    print(args)
    main(args)