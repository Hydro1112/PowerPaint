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
        is_validation=False,
        **kwargs
    ):
        self.data_root = data_root
        self.is_validation = is_validation

        with open(os.path.join(data_root, "metadata_run.json"), "r") as f:
            self.data = json.load(f)

        self.tokenizer = (
            pipe.tokenizer if hasattr(pipe, "tokenizer") else pipe
        )

        self.task_prompt = task_prompt

        self.image_transform = train_transforms

        mask_transforms_list = []
        for t in train_transforms.transforms:
            if isinstance(t, transforms.RandomResizedCrop):
                mask_transforms_list.append(transforms.RandomResizedCrop(
                    t.size, scale=t.scale, ratio=t.ratio,
                    interpolation=transforms.InterpolationMode.NEAREST
                ))
            elif isinstance(t, transforms.RandomHorizontalFlip):
                mask_transforms_list.append(transforms.RandomHorizontalFlip(p=t.p))
            elif isinstance(t, (transforms.Resize, transforms.CenterCrop, transforms.RandomCrop)):
                mask_transforms_list.append(t)
            else:
                break

        if not mask_transforms_list:
            mask_transforms_list = [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(512),
            ]

        self.mask_transform = transforms.Compose(mask_transforms_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        mask = Image.open(item["mask_path"]).convert("L")

        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        random.seed(seed)
        pixel_values = self.image_transform(image)

        torch.manual_seed(seed)
        random.seed(seed)
        mask_transformed = self.mask_transform(mask)

        mask_tensor = torch.from_numpy(
            np.array(mask_transformed) / 255.0
        ).unsqueeze(0).float()

        # Keep validation loss reproducible: training still samples the task at
        # random, while validation assigns each sample to one fixed task.
        task_key = (
            "text_guided_object_synthesis"
            if (self.is_validation and idx % 2 == 0) or (not self.is_validation and random.random() < 0.5)
            else "object_removal"
        )

        caption = (
            item["caption"]
            if task_key == "text_guided_object_synthesis"
            else ""
        )

        token = self.task_prompt[task_key].placeholder_tokens

        if isinstance(token, list):
            token = token[0]

        full_prompt = f"{caption} {token}".strip()

        input_ids = self._tokenize(full_prompt)
        input_idsA = self._tokenize(token)
        input_idsB = self._tokenize("")

        tradeoff_weight = random.uniform(0.5, 1.0)

        return {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "input_ids": input_ids,
            "input_idsA": input_idsA,
            "input_idsB": input_idsB,
            "tradeoff": torch.tensor([tradeoff_weight, 1.0 - tradeoff_weight])
        }

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
