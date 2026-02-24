import os
import random
import sys
import time
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    for existing_path in sys.path:
        if os.path.basename(existing_path) == "ComfyUI" and os.path.isdir(existing_path):
            return
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        # Silently try alternative import
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            # No extra config available, skip silently
            return

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    # File is optional, so no message if not found


# Only execute once at module level
_initialized = False
if not _initialized:
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    _initialized = True


def configure_local_paths() -> None:
    """Configure ComfyUI to use local models and output directories in facial_anonymisation folder"""
    import folder_paths
    
    # Get the facial_anonymisation directory path
    facial_anonymisation_dir = Path(__file__).parent
    models_dir = facial_anonymisation_dir / "models"
    output_dir = facial_anonymisation_dir / "output"
    
    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different model types
    (models_dir / "text_encoders").mkdir(exist_ok=True)
    (models_dir / "unet").mkdir(exist_ok=True)
    (models_dir / "vae").mkdir(exist_ok=True)
    (models_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Configure folder_paths to use local directories
    folder_paths.add_model_folder_path("text_encoders", str(models_dir / "text_encoders"))
    folder_paths.add_model_folder_path("unet", str(models_dir / "unet"))
    folder_paths.add_model_folder_path("vae", str(models_dir / "vae"))
    folder_paths.add_model_folder_path("checkpoints", str(models_dir / "checkpoints"))
    
    # Configure output directory
    folder_paths.set_output_directory(str(output_dir))
    
    print(f"✓ Models directory: {models_dir}")
    print(f"✓ Output directory: {output_dir}")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes (init_extra_nodes is async, so use loop.run_until_complete)
    loop.run_until_complete(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


def main():
    configure_local_paths()
    import_custom_nodes()
    
    # List of prompts for multiple generations
    prompts = [
        "Cinematic portrait of a futuristic cyberpunk woman, neon blue lighting, high detail skin texture, wearing reflective visor, bokeh background, 8k resolution, hyper-realistic.",
        "Professional portrait of a businessman in modern office, natural lighting, confident expression, sharp focus, 4k quality.",
        "Artistic portrait of a young woman with colorful paint splashes, creative studio lighting, vibrant colors, high detail."
    ]
    
    print("\n" + "="*60)
    print(f"   STARTING BATCH IMAGE GENERATION ({len(prompts)} images)")
    print("="*60)
    total_start_time = time.time()
    
    with torch.inference_mode():
        # Load models once (they will be reused for all generations)
        print("\n▶ Loading models...")
        models_load_start = time.time()
        
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_39 = cliploader.load_clip(
            clip_name="qwen_3_4b.safetensors", type="lumina2", device="default"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_40 = vaeloader.load_vae(vae_name="ae.safetensors")

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_46 = unetloader.load_unet(
            unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default"
        )
        
        models_load_time = time.time() - models_load_start
        print(f"✓ Models loaded in {models_load_time:.2f} seconds")

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        # Generate images for each prompt
        for idx, prompt_text in enumerate(prompts, start=1):
            print(f"\n{'='*60}")
            print(f"   GENERATING IMAGE {idx}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt_text[:80]}...")
            
            gen_start_time = time.time()
            
            emptylatentimage_41 = emptylatentimage.generate(
                width=1024, height=1024, batch_size=1
            )

            cliptextencode_45 = cliptextencode.encode(
                text=prompt_text,
                clip=get_value_at_index(cliploader_39, 0),
            )

            modelsamplingauraflow_47 = modelsamplingauraflow.patch_aura(
                shift=1, model=get_value_at_index(unetloader_46, 0)
            )

            conditioningzeroout_42 = conditioningzeroout.zero_out(
                conditioning=get_value_at_index(cliptextencode_45, 0)
            )

            ksampler_44 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=9,
                cfg=1,
                sampler_name="res_multistep",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(modelsamplingauraflow_47, 0),
                positive=get_value_at_index(cliptextencode_45, 0),
                negative=get_value_at_index(conditioningzeroout_42, 0),
                latent_image=get_value_at_index(emptylatentimage_41, 0),
            )

            vaedecode_43 = vaedecode.decode(
                samples=get_value_at_index(ksampler_44, 0),
                vae=get_value_at_index(vaeloader_40, 0),
            )

            saveimage_9 = saveimage.save_images(
                filename_prefix=f"z-image_{idx}", images=get_value_at_index(vaedecode_43, 0)
            )
            
            gen_time = time.time() - gen_start_time
            print(f"✓ Image {idx} generated in {gen_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    avg_time = total_time / len(prompts)
    
    print("\n" + "="*60)
    print("   BATCH GENERATION COMPLETED")
    print("="*60)
    print(f"✓ Total images: {len(prompts)}")
    print(f"✓ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"✓ Average time per image: {avg_time:.2f} seconds")
    print(f"✓ Models load time: {models_load_time:.2f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
