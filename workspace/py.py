import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

print("Starting pipeline...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

print("Loading image...")
# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
print("Generating video...")
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]


print("Exporting video...")
export_to_video(frames, "generated.mp4", fps=7)


print("Done!")