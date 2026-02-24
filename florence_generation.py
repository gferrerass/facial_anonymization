import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


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

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_1 = loadimage.load_image(image="0_Parade_marchingband_1_1048.jpg")

        downloadandloadflorence2model = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadFlorence2Model"
        ]()
        downloadandloadflorence2model_3 = downloadandloadflorence2model.loadmodel(
            model="MiaoshouAI/Florence-2-base-PromptGen-v1.5",
            precision="fp16",
            attention="sdpa",
            convert_to_safetensors=False,
        )

        florence2run = NODE_CLASS_MAPPINGS["Florence2Run"]()
        florence2run_21 = florence2run.encode(
            text_input="",
            task="more_detailed_caption",
            fill_mask=True,
            keep_model_loaded=False,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=True,
            output_mask_select="",
            seed=random.randint(1, 2**64),
            image=get_value_at_index(loadimage_1, 0),
            florence2_model=get_value_at_index(downloadandloadflorence2model_3, 0),
        )

        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_6 = cliploader.load_clip(
            clip_name="qwen_3_4b.safetensors", type="lumina2", device="default"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_4 = cliptextencode.encode(
            text=get_value_at_index(florence2run_21, 2),
            clip=get_value_at_index(cliploader_6, 0),
        )

        cliptextencode_5 = cliptextencode.encode(
            text="", clip=get_value_at_index(cliploader_6, 0)
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_7 = unetloader.load_unet(
            unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_12 = vaeloader.load_vae(vae_name="ae.safetensors")

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_23 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            modelsamplingauraflow_16 = modelsamplingauraflow.patch_aura(
                shift=6, model=get_value_at_index(unetloader_7, 0)
            )

            ksampler_20 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=9,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(modelsamplingauraflow_16, 0),
                positive=get_value_at_index(cliptextencode_4, 0),
                negative=get_value_at_index(cliptextencode_5, 0),
                latent_image=get_value_at_index(emptylatentimage_23, 0),
            )

            vaedecode_13 = vaedecode.decode(
                samples=get_value_at_index(ksampler_20, 0),
                vae=get_value_at_index(vaeloader_12, 0),
            )

            saveimage_22 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_13, 0)
            )


if __name__ == "__main__":
    main()
