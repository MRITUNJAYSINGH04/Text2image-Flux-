

This repo contains minimal inference code to run text-to-image and image-to-image with our Flux latent rectified flow transformers.

## Local installation

```bash
cd $HOME && git clone https://github.com/MRITUNJAYSINGH04/Text2image-Flux-.git
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
``

```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run

```bash
python -m flux --name <name> --loop
```

Or to generate a single sample run

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

```bash
streamlit run demo_st.py
```

We also offer a Gradio-based demo for an interactive experience. To run the Gradio demo:

```bash
python demo_gr.py --name flux-schnell --device cuda
```

Options:

- `--name`: Choose the model to use (options: "flux-schnell", "flux-dev")
- `--device`: Specify the device to use (default: "cuda" if available, otherwise "cpu")
- `--offload`: Offload model to CPU when not in use
- `--share`: Create a public link to your demo

To run the demo with the dev model and create a public link:

```bash
python demo_gr.py --name flux-dev --share
```

```python
import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")
```

