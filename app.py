import argparse
import os
import random

import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import (
    CLIPTextModel,
    DPTFeatureExtractor,
    DPTForDepthEstimation,
    MarianMTModel,
    MarianTokenizer,
)

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


torch.set_grad_enabled(False)

# Gradio converts a sketch's alpha channel to a white RGB mask before calling
# this app. Accept RGBA too, so direct callers follow the same convention.
MASK_DILATE_PX = 1


def prepare_editor_mask(mask, image_size, dilate_px=MASK_DILATE_PX):
    """Convert a Gradio sketch to the binary white mask expected by PowerPaint."""
    if mask is None:
        raise ValueError("Hãy tải ảnh và tô vùng cần chỉnh sửa trước khi chạy.")
    if "A" in mask.getbands() and mask.getchannel("A").getbbox() is not None:
        mask_l = mask.getchannel("A")
    else:
        mask_l = mask.convert("L")
    mask_l = mask_l.resize(image_size, Image.Resampling.NEAREST)
    mask_l = mask_l.point(lambda value: 255 if value > 127 else 0, mode="L")
    if dilate_px > 0:
        mask_l = mask_l.filter(ImageFilter.MaxFilter(size=dilate_px * 2 + 1))
    return mask_l


def compose_inpainted_result(base_image, generated, mask):
    """Apply the eval notebook's colour match, sharpening, and feathered blend."""
    base_image = base_image.convert("RGB")
    generated = generated.convert("RGB").resize(base_image.size, Image.Resampling.LANCZOS)
    hard_mask = mask.convert("L").resize(base_image.size, Image.Resampling.NEAREST)
    hard_mask = hard_mask.point(lambda value: 255 if value > 127 else 0, mode="L")

    outer = hard_mask.filter(ImageFilter.MaxFilter(size=25))
    mask_np = np.asarray(hard_mask, dtype=np.uint8) > 0
    ring_np = (np.asarray(outer, dtype=np.uint8) > 0) & (~mask_np)
    if ring_np.any():
        base_pixels = np.asarray(base_image, dtype=np.float32)
        generated_pixels = np.asarray(generated, dtype=np.float32)
        base_mean = base_pixels[ring_np].mean(axis=0)
        generated_mean = generated_pixels[ring_np].mean(axis=0)
        base_std = base_pixels[ring_np].std(axis=0) + 1e-6
        generated_std = generated_pixels[ring_np].std(axis=0) + 1e-6
        matched = (generated_pixels - generated_mean) * (base_std / generated_std) + base_mean
        generated = Image.fromarray(np.uint8(np.clip(matched, 0, 255)))

    generated = generated.filter(ImageFilter.UnsharpMask(radius=1.2, percent=100, threshold=2))
    alpha = hard_mask.filter(ImageFilter.GaussianBlur(radius=0.7))
    alpha_np = (np.asarray(alpha, dtype=np.float32) / 255.0)[..., None]
    output = np.asarray(base_image, dtype=np.float32) * (1.0 - alpha_np)
    output += np.asarray(generated, dtype=np.float32) * alpha_np
    return Image.fromarray(np.uint8(np.clip(output, 0, 255)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    """Object removal is prompt-free and uses the recommended CFG value."""
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only
        self.translation_model = None
        self.translation_tokenizer = None

        # initialize powerpaint pipeline
        if version == "ppt-v1":
            self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )

            # loading pre-trained weights
            load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"))
            load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"))
            self.pipe = self.pipe.to("cuda")

            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.control_pipe = controlnetPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            self.control_pipe = self.control_pipe.to("cuda")

            self.current_control = "canny"
            # controlnet_conditioning_scale = 0.8
        else:
            # brushnet-based version
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            text_encoder_brushnet = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            brushnet = BrushNetModel.from_unet(unet)
            base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                text_encoder_brushnet=text_encoder_brushnet,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=False,
                safety_checker=None,
            )
            self.pipe.unet = UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained=base_model_path,
                subfolder="tokenizer",
                revision=None,
                torch_type=weight_dtype,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder_brushnet,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )
            load_model(
                self.pipe.brushnet,
                os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
            )

            self.pipe.text_encoder_brushnet.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
            )

            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            self.pipe.enable_model_cpu_offload()
            self.pipe = self.pipe.to("cuda")

    def translate_vietnamese_prompt(self, prompt):
        """Translate Vietnamese on demand, keeping model start-up fast."""
        if not prompt or not prompt.strip():
            return ""

        if self.translation_model is None:
            model_name = "Helsinki-NLP/opus-mt-vi-en"
            try:
                self.translation_tokenizer = MarianTokenizer.from_pretrained(
                    model_name, local_files_only=self.local_files_only
                )
                self.translation_model = MarianMTModel.from_pretrained(
                    model_name, local_files_only=self.local_files_only
                ).eval()
            except OSError as exc:
                raise RuntimeError(
                    "Không tải được bộ dịch Việt–Anh. Hãy kết nối Internet một lần để tải "
                    "Helsinki-NLP/opus-mt-vi-en, hoặc dùng Prompt tiếng Anh."
                ) from exc

        encoded = self.translation_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        with torch.no_grad():
            translated = self.translation_model.generate(**encoded, max_new_tokens=256)
        return self.translation_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    def resolve_prompt(self, prompt, language="Tự động"):
        """Resolve a bilingual prompt, including Vietnamese written without accents."""
        prompt = (prompt or "").strip()
        if language == "Tiếng Việt":
            return self.translate_vietnamese_prompt(prompt)
        if language == "English":
            return prompt
        if any(
            character in "ăâđêôơưĂÂĐÊÔƠƯ" or "\u1ea0" <= character <= "\u1ef9" for character in prompt
        ):
            return self.translate_vietnamese_prompt(prompt)
        return prompt

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def load_controlnet(self, control_type):
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = control_type

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
    ):
        input_image["image"] = input_image["image"].convert("RGB")
        input_image["mask"] = prepare_editor_mask(input_image["mask"], input_image["image"].size)
        size1, size2 = input_image["image"].size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W), Image.Resampling.LANCZOS)
        input_image["mask"] = input_image["mask"].resize((H, W), Image.Resampling.NEAREST)
        base_image = input_image["image"].convert("RGB")
        set_seed(seed)

        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=base_image,
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]
        else:
            # for brushnet-based method
            np_inpimg = np.asarray(base_image, dtype=np.float32)
            np_inmask = np.asarray(input_image["mask"], dtype=np.float32) / 255.0
            np_inpimg = np_inpimg * (1 - np_inmask[..., None])
            masked_image = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=masked_image,
                mask=input_image["mask"].convert("RGB"),
                num_inference_steps=ddim_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
            ).images[0]

        final_result = compose_inpainted_result(base_image, result, input_image["mask"])
        mask_np = np.array(input_image["mask"].convert("RGB"))
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )
        dict_res = [input_image["mask"].convert("RGB"), result_m]
        dict_out = [final_result]
        return dict_out, dict_res

    def predict_controlnet(
        self,
        input_image,
        input_control_image,
        control_type,
        prompt,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        controlnet_conditioning_scale,
    ):
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
        input_image["image"] = input_image["image"].convert("RGB")
        input_image["mask"] = prepare_editor_mask(input_image["mask"], input_image["image"].size)
        size1, size2 = input_image["image"].size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W), Image.Resampling.LANCZOS)
        input_image["mask"] = input_image["mask"].resize((H, W), Image.Resampling.NEAREST)
        base_image = input_image["image"].convert("RGB")

        if control_type != self.current_control:
            self.load_controlnet(control_type)
        controlnet_image = input_control_image
        if control_type == "canny":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = np.array(controlnet_image)
            controlnet_image = cv2.Canny(controlnet_image, 100, 200)
            controlnet_image = controlnet_image[:, :, None]
            controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
            controlnet_image = Image.fromarray(controlnet_image)
        elif control_type == "pose":
            controlnet_image = self.openpose(controlnet_image)
        elif control_type == "depth":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = self.get_depth_map(controlnet_image)
        else:
            controlnet_image = self.hed(controlnet_image)

        mask_np = np.array(input_image["mask"].convert("RGB"))
        controlnet_image = controlnet_image.resize((H, W))
        set_seed(seed)
        result = self.control_pipe(
            promptA=promptB,
            promptB=promptA,
            tradoff=1.0,
            tradoff_nag=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=base_image,
            mask=input_image["mask"].convert("RGB"),
            control_image=controlnet_image,
            width=H,
            height=W,
            guidance_scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=ddim_steps,
        ).images[0]
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        final_result = compose_inpainted_result(base_image, result, input_image["mask"])
        return [base_image, final_result], [controlnet_image, result_m]

    def infer(
        self,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        text_guided_prompt_language,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        shape_guided_prompt_language,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        outpaint_prompt_language,
        removal_prompt,
        removal_negative_prompt,
        enable_control=False,
        input_control_image=None,
        control_type="canny",
        controlnet_conditioning_scale=None,
    ):
        if task == "text-guided":
            prompt = self.resolve_prompt(text_guided_prompt, text_guided_prompt_language)
            negative_prompt = self.resolve_prompt(text_guided_negative_prompt, text_guided_prompt_language)
        elif task == "shape-guided":
            prompt = self.resolve_prompt(shape_guided_prompt, shape_guided_prompt_language)
            negative_prompt = self.resolve_prompt(shape_guided_negative_prompt, shape_guided_prompt_language)
        elif task == "object-removal":
            # Object removal is deliberately prompt-free.  A higher CFG value
            # makes the model favour a clean continuation of the background.
            prompt = ""
            negative_prompt = ""
            scale = 12
        elif task == "image-outpainting":
            prompt = self.resolve_prompt(outpaint_prompt, outpaint_prompt_language)
            negative_prompt = self.resolve_prompt(outpaint_negative_prompt, outpaint_prompt_language)
            return self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = self.resolve_prompt(text_guided_prompt, text_guided_prompt_language)
            negative_prompt = self.resolve_prompt(text_guided_negative_prompt, text_guided_prompt_language)

        # currently, we only support controlnet in PowerPaint-v1
        if self.version == "ppt-v1" and enable_control and task == "text-guided":
            return self.predict_controlnet(
                input_image,
                input_control_image,
                control_type,
                prompt,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                controlnet_conditioning_scale,
            )
        else:
            return self.predict(
                input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ppt-v2")
    args.add_argument("--version", type=str, default="ppt-v2")
    args.add_argument("--share", action="store_true")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument("--port", type=int, default=7860)
    args = args.parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaintController(weight_dtype, args.checkpoint_dir, args.local_files_only, args.version)

    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path, encoding="utf-8") as css_file:
        custom_css = css_file.read()
    prompt_guide_html = (
        "<div class='inline-prompt-guide'><b>💡 Prompt guide:</b> <b>Positive Prompt</b> mô tả điều muốn tạo; "
        "<b>Negative Prompt</b> mô tả điều cần tránh (ví dụ: mờ, chữ, watermark). "
        "Chọn <b>Tiếng Việt</b> nếu bạn nhập tiếng Việt có hoặc không dấu.</div>"
    )

    # ui
    with gr.Blocks(css=custom_css, title="PowerPaint") as demo:
        gr.HTML(
            "<header class='app-header'>"
            "<div class='brand-mark'>✦</div><div class='brand-copy'><h1>PowerPaint</h1>"
            "<p>High-Quality Versatile Image Inpainting</p></div>"
            "</header>"
        )
        gr.HTML(
            "<div class='notice'>ⓘ &nbsp;<b>Lưu ý:</b> Nếu kết quả chưa như mong muốn, hãy thử đổi tác vụ "
            "hoặc điều chỉnh prompt và Guidance Scale.</div>"
        )
        with gr.Row(equal_height=False, elem_classes=["workspace"]):
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                gr.HTML("<div class='section-title'>🖼️ &nbsp;Ảnh đầu vào</div>", elem_classes=["section-title-wrap"])
                with gr.Group(elem_classes=["panel-card", "input-card"]):
                    input_image = gr.Image(
                        source="upload",
                        tool="sketch",
                        type="pil",
                        show_label=False,
                        height=360,
                        brush_radius=None,
                        elem_id="input-image",
                    )

                task = gr.Radio(
                    ["text-guided", "object-removal", "shape-guided", "image-outpainting"],
                    value="shape-guided",
                    show_label=False,
                    visible=False,
                )

                with gr.Tabs(elem_id="task-tabs"):
                    with gr.Tab("✧  Shape-guided") as tab_shape_guided:
                        with gr.Group(elem_classes=["task-card"]):
                            enable_shape_guided = gr.Checkbox(
                                label="Bật Shape Guided Inpainting", value=True, interactive=False
                            )
                            shape_guided_prompt = gr.Textbox(
                                label="Positive Prompt",
                                placeholder="Ví dụ: một chú mèo trắng / a white cat",
                            )
                            shape_guided_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Ví dụ: mờ, chữ, watermark / blurry, text, watermark",
                            )
                            gr.HTML(prompt_guide_html)
                            shape_guided_prompt_language = gr.Dropdown(
                                ["Tự động", "Tiếng Việt", "English"],
                                label="Ngôn ngữ prompt (áp dụng cho cả 2 ô)",
                                value="Tự động",
                            )
                            fitting_degree = gr.Slider(
                                label="Blending Degree (độ khớp mask)", minimum=0, maximum=1, step=0.05, value=1
                            )
                    tab_shape_guided.select(fn=select_tab_shape_guided, inputs=None, outputs=task)

                    with gr.Tab("🗑  Removal") as tab_object_removal:
                        with gr.Group(elem_classes=["task-card", "removal-card"]):
                            gr.Markdown(
                                "#### Xoá đối tượng không cần prompt\n"
                                "Tô mask lên vật thể cần xoá. Guidance Scale tự động đặt là **12**."
                            )

                    with gr.Tab("▣  Outpainting") as tab_image_outpainting:
                        with gr.Group(elem_classes=["task-card"]):
                            outpaint_prompt = gr.Textbox(
                                label="Positive Prompt",
                                placeholder="Ví dụ: bãi biển lúc hoàng hôn / beach at sunset",
                            )
                            outpaint_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Ví dụ: mờ, chữ / blurry, text",
                            )
                            gr.HTML(prompt_guide_html)
                            outpaint_prompt_language = gr.Dropdown(
                                ["Tự động", "Tiếng Việt", "English"],
                                label="Ngôn ngữ prompt (áp dụng cho cả 2 ô)",
                                value="Tự động",
                            )
                            with gr.Row():
                                horizontal_expansion_ratio = gr.Slider(
                                    label="Mở rộng ngang", minimum=1, maximum=4, step=0.05, value=1
                                )
                                vertical_expansion_ratio = gr.Slider(
                                    label="Mở rộng dọc", minimum=1, maximum=4, step=0.05, value=1
                                )
                    tab_image_outpainting.select(fn=select_tab_image_outpainting, inputs=None, outputs=task)

                    with gr.Tab("✎  Text-guided") as tab_text_guided:
                        with gr.Group(elem_classes=["task-card"]):
                            text_guided_prompt = gr.Textbox(
                                label="Positive Prompt",
                                placeholder="Ví dụ: một chiếc xe đạp đỏ / a red bicycle",
                            )
                            text_guided_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Ví dụ: mờ, méo hình / blurry, distorted",
                            )
                            gr.HTML(prompt_guide_html)
                            text_guided_prompt_language = gr.Dropdown(
                                ["Tự động", "Tiếng Việt", "English"],
                                label="Ngôn ngữ prompt (áp dụng cho cả 2 ô)",
                                value="Tự động",
                            )
                            if args.version == "ppt-v1":
                                with gr.Accordion("Cài đặt ControlNet", open=False):
                                    enable_control = gr.Checkbox(label="Bật ControlNet")
                                    controlnet_conditioning_scale = gr.Slider(
                                        label="ControlNet conditioning scale", minimum=0, maximum=1, step=0.05, value=0.5
                                    )
                                    control_type = gr.Radio(["canny", "pose", "depth", "hed"], label="Control type", value="canny")
                                    input_control_image = gr.Image(source="upload", type="pil", label="Control image")
                    tab_text_guided.select(fn=select_tab_text_guided, inputs=None, outputs=task)

                run_button = gr.Button("✦  Chạy Inpainting", elem_id="run-button")
                with gr.Accordion("⚙ &nbsp;Tuỳ chọn nâng cao", open=False, elem_classes=["advanced-card"]):
                    ddim_steps = gr.Slider(label="Steps (số bước)", minimum=1, maximum=50, value=45, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale (độ bám prompt)", minimum=0.1, maximum=30.0, value=12, step=0.1,
                        elem_id="guidance-scale",
                    )
                    seed = gr.Slider(label="Seed (hạt giống)", minimum=0, maximum=2147483647, step=1, randomize=True)

            with gr.Column(scale=1, elem_classes=["right-panel"]):
                gr.HTML("<div class='section-title'>✧ &nbsp;Kết quả Inpainting</div>", elem_classes=["section-title-wrap"])
                with gr.Group(elem_classes=["panel-card", "output-card"]):
                    inpaint_result = gr.Gallery(
                        label="Generated images", show_label=False, columns=1, elem_id="result-gallery"
                    )
                gr.HTML(
                    "<div class='section-title mask-section-title'>◉ &nbsp;Mask (vùng chỉnh sửa)</div>",
                    elem_classes=["section-title-wrap"],
                )
                with gr.Group(elem_classes=["panel-card", "mask-card"]):
                    gallery = gr.Gallery(label="Generated masks", show_label=False, columns=1, elem_id="mask-gallery")

        tab_object_removal.select(
            fn=select_tab_object_removal,
            inputs=None,
            outputs=task,
            _js="""() => {
                document.querySelectorAll('#guidance-scale input').forEach((input) => {
                    input.value = 12;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                });
                return [];
            }""",
        )
        gr.HTML("<div class='tip'>💡 &nbsp;Mẹo: Tô mask chính xác quanh vùng cần chỉnh để có kết quả tốt hơn.</div>")

        # Keep the prompt inputs in the inference signature without rendering
        # them: Object Removal never accepts or sends prompt text.
        removal_prompt = gr.State("")
        removal_negative_prompt = gr.State("")

        if args.version == "ppt-v1":
            run_button.click(
                fn=controller.infer,
                inputs=[
                    input_image,
                    text_guided_prompt,
                    text_guided_negative_prompt,
                    text_guided_prompt_language,
                    shape_guided_prompt,
                    shape_guided_negative_prompt,
                    shape_guided_prompt_language,
                    fitting_degree,
                    ddim_steps,
                    scale,
                    seed,
                    task,
                    vertical_expansion_ratio,
                    horizontal_expansion_ratio,
                    outpaint_prompt,
                    outpaint_negative_prompt,
                    outpaint_prompt_language,
                    removal_prompt,
                    removal_negative_prompt,
                    enable_control,
                    input_control_image,
                    control_type,
                    controlnet_conditioning_scale,
                ],
                outputs=[inpaint_result, gallery],
            )
        else:
            run_button.click(
                fn=controller.infer,
                inputs=[
                    input_image,
                    text_guided_prompt,
                    text_guided_negative_prompt,
                    text_guided_prompt_language,
                    shape_guided_prompt,
                    shape_guided_negative_prompt,
                    shape_guided_prompt_language,
                    fitting_degree,
                    ddim_steps,
                    scale,
                    seed,
                    task,
                    vertical_expansion_ratio,
                    horizontal_expansion_ratio,
                    outpaint_prompt,
                    outpaint_negative_prompt,
                    outpaint_prompt_language,
                    removal_prompt,
                    removal_negative_prompt,
                ],
                outputs=[inpaint_result, gallery],
            )

    demo.queue()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
