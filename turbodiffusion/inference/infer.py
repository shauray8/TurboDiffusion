import argparse
import math
import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


class VideoGenerationPipeline:
    """Pipeline that loads models once and handles multiple inference requests."""
    
    def __init__(self, args):
        """Initialize pipeline with models loaded once."""
        self.args = args
        
        log.info("Initializing Video Generation Pipeline...")
        
        # Load DiT models once
        log.info(f"Loading DiT models.")
        self.high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args).cpu()
        torch.cuda.empty_cache()
        self.low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args).cpu()
        torch.cuda.empty_cache()
        log.success(f"Successfully loaded DiT model.")

        log.info("Setting up torch.compile for both models...")
        self.high_noise_model = torch.compile(self.high_noise_model, mode="max-autotune-no-cudagraphs", dynamic=True)
        self.low_noise_model = torch.compile(self.low_noise_model, mode="max-autotune-no-cudagraphs", dynamic=True)
        log.success("Models prepared for compilation (will compile on first forward pass).")

        # Load tokenizer once
        self.tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
        
        log.success("Pipeline initialization complete!")
    
    def generate(
        self,
        prompt: str,
        image_path: str,
        save_path: str,
        num_samples: int = None,
        num_frames: int = None,
        num_steps: int = None,
        resolution: str = None,
        aspect_ratio: str = None,
        adaptive_resolution: bool = None,
        seed: int = None,
        sigma_max: float = None,
        boundary: float = None,
        ode: bool = None,
    ):
        """Generate video - uses args defaults if parameters not provided."""
        # Use provided values or fall back to args
        num_samples = num_samples if num_samples is not None else self.args.num_samples
        num_frames = num_frames if num_frames is not None else self.args.num_frames
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        resolution = resolution if resolution is not None else self.args.resolution
        aspect_ratio = aspect_ratio if aspect_ratio is not None else self.args.aspect_ratio
        adaptive_resolution = adaptive_resolution if adaptive_resolution is not None else self.args.adaptive_resolution
        seed = seed if seed is not None else self.args.seed
        sigma_max = sigma_max if sigma_max is not None else self.args.sigma_max
        boundary = boundary if boundary is not None else self.args.boundary
        ode = ode if ode is not None else self.args.ode
        
        log.info(f"Computing embedding for prompt: {prompt}")
        text_emb = get_umt5_embedding(checkpoint_path=self.args.text_encoder_path, prompts=prompt).to(**tensor_kwargs)
        clear_umt5_memory()

        log.info(f"Loading and preprocessing image from: {image_path}")
        input_image = Image.open(image_path).convert("RGB")
        if adaptive_resolution:
            log.info("Adaptive resolution mode enabled.")
            base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
            max_resolution_area = base_w * base_h
            log.info(f"Target area is based on {resolution} {aspect_ratio} (~{max_resolution_area} pixels).")

            orig_w, orig_h = input_image.size
            image_aspect_ratio = orig_h / orig_w

            ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
            ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

            stride = self.tokenizer.spatial_compression_factor * 2
            lat_h = round(ideal_h / stride)
            lat_w = round(ideal_w / stride)
            h = lat_h * stride
            w = lat_w * stride

            log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
        else:
            log.info("Fixed resolution mode.")
            w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
            log.info(f"Resolution set to: {w}x{h}")
        F = num_frames
        lat_h = h // self.tokenizer.spatial_compression_factor
        lat_w = w // self.tokenizer.spatial_compression_factor
        lat_t = self.tokenizer.get_latent_num_frames(F)

        log.info(f"Preprocessing image to {w}x{h}...")
        image_transforms = T.Compose(
            [
                T.ToImage(),
                T.Resize(size=(h, w), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

        with torch.no_grad():
            frames_to_encode = torch.cat(
                [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
            )  # -> B, C, T, H, W
            encoded_latents = self.tokenizer.encode(frames_to_encode)  # -> B, C_lat, T_lat, H_lat, W_lat

        msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
        msk[:, :, 0, :, :] = 1.0

        y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
        y = y.repeat(num_samples, 1, 1, 1, 1)

        log.info(f"Generating with prompt: {prompt}")
        condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=num_samples), "y_B_C_T_H_W": y}

        to_show = []

        state_shape = [self.tokenizer.latent_ch, lat_t, lat_h, lat_w]

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(seed)

        init_noise = torch.randn(
            num_samples,
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        mid_t = [1.5, 1.4, 1.0][:num_steps - 1]

        t_steps = torch.tensor(
            [math.atan(sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )

        # Convert TrigFlow timesteps to RectifiedFlow
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        total_steps = t_steps.shape[0] - 1
        self.high_noise_model.cuda()
        net = self.high_noise_model
        switched = False
        for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            if t_cur.item() < boundary and not switched:
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
                self.low_noise_model.cuda()
                net = self.low_noise_model
                switched = True
                log.info("Switched to low noise model.")
            with torch.no_grad():
                v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                    torch.float64
                )
                if ode:
                    x = x - (t_cur - t_next) * v_pred
                else:
                    x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                        *x.shape,
                        dtype=torch.float32,
                        device=tensor_kwargs["device"],
                        generator=generator,
                    )
        samples = x.float()

        video = self.tokenizer.decode(samples)

        to_show.append(video.float().cpu())

        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

        save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), save_path, fps=16)
        
        # Clean up GPU memory after generation
        self.high_noise_model.cpu()
        self.low_noise_model.cpu()
        del text_emb, image_tensor, frames_to_encode, encoded_latents, msk, y, condition
        del init_noise, t_steps, x, ones, v_pred, samples, video, to_show
        torch.cuda.empty_cache()
        
        log.success(f"Video saved to: {save_path}")
        return save_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.2 I2V with High/Low Noise models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image for I2V generation")
    parser.add_argument("--high_noise_model_path", type=str, required=True, help="Path to the high-noise model")
    parser.add_argument("--low_noise_model_path", type=str, required=True, help="Path to the low-noise model")
    parser.add_argument("--boundary", type=float, default=0.9, help="Timestep boundary for switching from high to low noise model")
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=200, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=77, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--resolution", default="720p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true", help="If set, adapts the output resolution to the input image's aspect ratio, using the area defined by --resolution and --aspect_ratio as a target.")
    parser.add_argument("--ode", action="store_true", help="Use ODE for sampling (sharper but less robust than SDE)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    pipeline = VideoGenerationPipeline(args)
    
    pipeline.generate(
        prompt=args.prompt,
        image_path=args.image_path,
        save_path=args.save_path,
    )
    pipeline.generate(
        prompt=args.prompt,
        image_path=args.image_path,
        save_path=args.save_path,
    )
    
