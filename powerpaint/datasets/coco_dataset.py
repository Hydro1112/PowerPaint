import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_transforms,
        pipe,
        task_prompt,
        data_root,
        prob=1.0,
        **kwargs
    ):
        self.data_root = data_root

        with open(os.path.join(data_root, "metadata_run.json"), "r") as f:
            self.data = json.load(f)

        self.tokenizer = (
            pipe.tokenizer if hasattr(pipe, "tokenizer") else pipe
        )

        self.task_prompt = task_prompt

        self.image_transform = transforms.Compose([
            transforms.Resize(
                512,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                512,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.CenterCrop(512)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        mask = Image.open(item["mask_path"]).convert("L")

        pixel_values = self.image_transform(image)

        mask_transformed = self.mask_transform(mask)

        mask_tensor = torch.from_numpy(
            np.array(mask_transformed) / 255.0
        ).unsqueeze(0).float()

        task_key = (
            "object_inpainting"
            if random.random() < 0.5
            else "context_aware"
        )

        caption = (
            item["caption"]
            if task_key == "object_inpainting"
            else ""
        )

        token = self.task_prompt[task_key].placeholder_tokens

        if isinstance(token, list):
            token = token[0]

        full_prompt = f"{caption} {token}".strip()

        input_ids = self._tokenize(full_prompt)
        input_idsA = self._tokenize(token)
        input_idsB = self._tokenize("")

        return {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "input_ids": input_ids,
            "input_idsA": input_idsA,
            "input_idsB": input_idsB,
            "tradeoff": torch.tensor([1.0, 0.0])
        }

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]