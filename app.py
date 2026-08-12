import argparse
import os
import random
from pyngrok import ngrok

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    CLIPTextModel,
)

from diffusers import UniPCMultistepScheduler
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.content_filter import NSFWImageFilter, prompt_is_blocked
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


torch.set_grad_enabled(False)

# Gradio converts a sketch's alpha channel to a white RGB mask before calling
# this app. Accept RGBA too, so direct callers follow the same convention.


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type):
    # The brushnet pipeline only needs the prompt in `promptU`; keeping the
    # prompt and negative prompt out of promptA/promptB matches the reference
    # implementation and avoids over-conditioning that shifts output colours.
    if control_type == "object-removal" or control_type == "image-outpainting":
        promptA = " P_ctxt"
        promptB = " P_ctxt"
        negative_promptA = " P_obj"
        negative_promptB = " P_obj"
    elif control_type == "shape-guided":
        promptA = " P_shape"
        promptB = " P_ctxt"
        negative_promptA = "P_shape"
        negative_promptB = "P_ctxt"
    else:
        promptA = " P_obj"
        promptB = " P_obj"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, enable_nsfw_filter=True) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only
        self.weight_dtype = weight_dtype
        self.translation_model = None
        self.translation_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nsfw_filter = NSFWImageFilter(device=self.device, weight_dtype=weight_dtype) if enable_nsfw_filter else None

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
        # Do not move the whole pipeline to CUDA after enabling CPU offload.
        # The offload hook moves each component to the GPU only when needed;
        # calling .to("cuda") here defeats that behavior and can exhaust Colab VRAM.

        self.load_translation_model()

    def load_translation_model(self):
        """Load the Vietnamese→English translation model up front."""
        if self.translation_model is not None:
            return
        model_name = "facebook/nllb-200-distilled-600M"
        try:
            self.translation_tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=self.local_files_only
            )
            self.translation_tokenizer.src_lang = "vie_Latn"
            self.translation_tokenizer.tgt_lang = "eng_Latn"
            model_kwargs = {}
            if self.weight_dtype == torch.float16 and self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, local_files_only=self.local_files_only, **model_kwargs
            ).eval()
            self.translation_model = self.translation_model.to(self.device)
        except OSError as exc:
            raise RuntimeError(
                "Không tải được bộ dịch Việt–Anh. Hãy kết nối Internet một lần để tải "
                "facebook/nllb-200-distilled-600M, hoặc dùng Prompt tiếng Anh."
            ) from exc

    def translate_vietnamese_prompt(self, prompt):
        """Translate Vietnamese with the preloaded NLLB model."""
        if not prompt or not prompt.strip():
            return ""

        encoded = self.translation_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)
        with torch.no_grad():
            translated = self.translation_model.generate(
                **encoded,
                max_new_tokens=256,
                forced_bos_token_id=self.translation_tokenizer.convert_tokens_to_ids("eng_Latn"),
            )
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
        if input_image["mask"] is None:
            raise ValueError("Hãy tải ảnh và tô vùng cần chỉnh sửa trước khi chạy.")
        input_image["mask"] = input_image["mask"].convert("RGB")
        if self.nsfw_filter is not None and self.nsfw_filter.check_image(input_image["image"]):
            raise gr.Error(
                "Ảnh đầu vào chứa nội dung nhạy cảm (18+) và không được phép. "
                "Vui lòng thử lại với ảnh khác."
            )
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

        if task == "image-outpainting":
            prompt = prompt + " empty scene"
        if task == "object-removal":
            prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)

        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0
        np_inpimg = np_inpimg * (1 - np_inmask)
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

        if self.nsfw_filter is not None and self.nsfw_filter.check_image(result):
            raise gr.Error(
                "Kết quả tạo ra chứa nội dung nhạy cảm (18+) và không được phép. "
                "Vui lòng thử lại với prompt hoặc ảnh khác."
            )

        final_result = result
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
        removal_prompt_language,
        enable_shape_guided=True,
    ):
        # Check the raw (possibly Vietnamese) prompts before translation, so
        # 18+ wording is caught even if the NLLB translation rephrases it.
        for raw_prompt in (
            text_guided_prompt,
            shape_guided_prompt,
            outpaint_prompt,
            removal_prompt,
            text_guided_negative_prompt,
            shape_guided_negative_prompt,
            outpaint_negative_prompt,
            removal_negative_prompt,
        ):
            self._raise_if_blocked_prompt(raw_prompt)

        if task == "text-guided":
            prompt = self.resolve_prompt(text_guided_prompt, text_guided_prompt_language)
            negative_prompt = self.resolve_prompt(text_guided_negative_prompt, text_guided_prompt_language)
        elif task == "shape-guided":
            prompt = self.resolve_prompt(shape_guided_prompt, shape_guided_prompt_language)
            negative_prompt = self.resolve_prompt(shape_guided_negative_prompt, shape_guided_prompt_language)
            if not enable_shape_guided:
                task = "text-guided"
        elif task == "object-removal":
            prompt = self.resolve_prompt(removal_prompt, removal_prompt_language)
            negative_prompt = self.resolve_prompt(removal_negative_prompt, removal_prompt_language)
        elif task == "image-outpainting":
            prompt = self.resolve_prompt(outpaint_prompt, outpaint_prompt_language)
            negative_prompt = self.resolve_prompt(outpaint_negative_prompt, outpaint_prompt_language)
            self._raise_if_blocked_prompt(prompt)
            self._raise_if_blocked_prompt(negative_prompt)
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

        self._raise_if_blocked_prompt(prompt)
        self._raise_if_blocked_prompt(negative_prompt)
        return self.predict(
            input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
        )

    def _raise_if_blocked_prompt(self, prompt):
        """Raise a user-visible error when the resolved prompt is 18+."""
        if self.nsfw_filter is None:
            return
        if prompt_is_blocked(prompt):
            raise gr.Error(
                "Prompt chứa nội dung nhạy cảm (18+) và không được phép. "
                "Vui lòng thử lại với mô tả khác."
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ppt-v2")
    args.add_argument("--share", action="store_true")
    args.add_argument("--ngrok", action="store_true")
    args.add_argument("--ngrok_token", type=str, default="")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument(
        "--no-nsfw-filter",
        action="store_true",
        help="disable the 18+ filter (prompt keywords and image classifier)",
    )
    args.add_argument("--port", type=int, default=7860)
    args = args.parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaintController(
        weight_dtype, args.checkpoint_dir, args.local_files_only, enable_nsfw_filter=not args.no_nsfw_filter
    )

    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path, encoding="utf-8") as css_file:
        custom_css = css_file.read()
    prompt_guide_html = (
        "<div class='inline-prompt-guide'><b>Prompt guide:</b> <b>Positive Prompt</b> mô tả điều muốn tạo; "
        "<b>Negative Prompt</b> mô tả điều cần tránh (ví dụ: mờ, chữ, watermark). "
        "Chọn <b>Tiếng Việt</b> nếu bạn nhập tiếng Việt có hoặc không dấu.</div>"
    )

    # ui
    with gr.Blocks(css=custom_css, title="PowerPaint") as demo:
        gr.HTML(
            "<header class='app-header'><h1>PowerPaint</h1></header>"
        )
        gr.HTML(
            "<div class='notice'><b>Lưu ý:</b> Nếu kết quả chưa như mong muốn, hãy thử đổi tác vụ "
            "hoặc điều chỉnh prompt và Guidance Scale.</div>"
        )
        with gr.Row(equal_height=False, elem_classes=["workspace"]):
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                gr.HTML("<div class='section-title'>Ảnh đầu vào</div>", elem_classes=["section-title-wrap"])
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
                    with gr.Tab("Shape-guided") as tab_shape_guided:
                        with gr.Group(elem_classes=["task-card"]):
                            enable_shape_guided = gr.Checkbox(
                                label="Bật Shape Guided Inpainting", value=True, interactive=True
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

                    with gr.Tab("Removal") as tab_object_removal:
                        with gr.Group(elem_classes=["task-card", "removal-card"]):
                            gr.Markdown(
                                "#### Xoá đối tượng\n"
                                "Tô mask lên vật thể cần xoá. Prompt mô tả phần nền sẽ được lấp vào."
                            )
                            removal_prompt = gr.Textbox(
                                label="Positive Prompt",
                                placeholder="Ví dụ: bãi biển vắng người / empty beach",
                            )
                            removal_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Ví dụ: mờ, chữ / blurry, text",
                            )
                            gr.HTML(prompt_guide_html)
                            removal_prompt_language = gr.Dropdown(
                                ["Tự động", "Tiếng Việt", "English"],
                                label="Ngôn ngữ prompt (áp dụng cho cả 2 ô)",
                                value="Tự động",
                            )

                    with gr.Tab("Outpainting") as tab_image_outpainting:
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

                    with gr.Tab("Text-guided") as tab_text_guided:
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
                    tab_text_guided.select(fn=select_tab_text_guided, inputs=None, outputs=task)

                run_button = gr.Button("Chạy Inpainting", elem_id="run-button")
                with gr.Accordion("Tuỳ chọn nâng cao", open=False, elem_classes=["advanced-card"]):
                    ddim_steps = gr.Slider(label="Steps (số bước)", minimum=1, maximum=50, value=45, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale (độ bám prompt)", minimum=0.1, maximum=30.0, value=7.5, step=0.1,
                        elem_id="guidance-scale",
                    )
                    seed = gr.Slider(label="Seed (hạt giống)", minimum=0, maximum=2147483647, step=1, randomize=True)

            with gr.Column(scale=1, elem_classes=["right-panel"]):
                gr.HTML("<div class='section-title'>Kết quả Inpainting</div>", elem_classes=["section-title-wrap"])
                with gr.Group(elem_classes=["panel-card", "output-card"]):
                    inpaint_result = gr.Gallery(
                        label="Generated images", show_label=False, columns=1, elem_id="result-gallery"
                    )
                gr.HTML(
                    "<div class='section-title mask-section-title'>Mask (vùng chỉnh sửa)</div>",
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
                    input.value = 7.5;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                });
                return [];
            }""",
        )
        gr.HTML("<div class='tip'>Mẹo: Tô mask chính xác quanh vùng cần chỉnh để có kết quả tốt hơn.</div>")

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
                removal_prompt_language,
                enable_shape_guided,
            ],
            outputs=[inpaint_result, gallery],
        )

    demo.queue()
    if args.ngrok:
        token = args.ngrok_token or os.getenv("NGROK_AUTHTOKEN")

        if token:
            ngrok.set_auth_token(token)
        else:
            raise RuntimeError(
                "Không tìm thấy NGROK_AUTHTOKEN. "
                "Hãy đặt biến môi trường hoặc truyền --ngrok_token."
            )

        tunnel = ngrok.connect(args.port)

        print("=" * 60)
        print("Ngrok URL:")
        print(tunnel.public_url)
        print("=" * 60)
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
