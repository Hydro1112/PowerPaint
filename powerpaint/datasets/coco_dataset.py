import os
import io
import json
import sys
import torch
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms


# Ngưỡng diện tích tối thiểu (tính theo px) để bỏ các annotation quá nhỏ.
MIN_ANN_AREA = 1024

# Số object tối đa được random chọn để hợp mask trong một sample.
# Tránh việc chọn quá nhiều object -> mask phủ gần hết ảnh (giống pipeline cũ).
MAX_OBJECTS_PER_SAMPLE = 3

# Số pixel mask tối thiểu (tính trên mask SAU khi RandomCrop/Resize/Flip).
# Dưới ngưỡng này thì mask quá nhỏ để model học -> thử subset object khác.
MIN_MASK_PIXELS = 300

# Số lần thử sinh mask hợp lệ trước khi chấp nhận mask cuối cùng (dù nhỏ).
MAX_MASK_ATTEMPTS = 8

# Mask không được phủ quá tỉ lệ diện tích ảnh (tránh mask chiếm gần hết ảnh).
MAX_MASK_RATIO = 0.4

# Khoảng random affine cho mask lấy từ ảnh khác (object removal).
CROSS_MASK_SCALE = (0.3, 1.0)       # tỉ lệ scale ngẫu nhiên
CROSS_MASK_ROTATION = (-20, 20)     # góc xoay (độ)
CROSS_MASK_TRANSLATION = 0.2        # tỉ lệ dịch chuyển so với kích thước ảnh

# Số lần thử tối đa tìm ảnh khác có annotation hợp lệ.
MAX_CROSS_MASK_TRIES = 20

# Annotation gốc của COCO2017 thường được giải nén ở nơi khác với data_root (Colab).
# Fallback tìm annotation trong data_root, rồi các thư mục annotations quen thuộc.
_ANNOTATION_FALLBACK_DIRS = [
    "/content/coco_temp/annotations",
    "/content/coco_clean_splits/annotations",
]


def _silent_coco(*args, **kwargs):
    """Chặn log 'loading annotations into memory...' của pycocotools."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return COCO(*args, **kwargs)
    finally:
        sys.stdout = old_stdout


def _resolve_annotation_path(data_root, annotation_file):
    """Trả về đường dẫn annotation file thật sự, hỗ trợ cả path tuyệt đối và tên file."""
    if annotation_file is None:
        annotation_file = "instances_train2017.json"
    candidates = []
    if os.path.isabs(annotation_file):
        candidates.append(annotation_file)
    else:
        candidates.append(os.path.join(data_root, annotation_file))
        for d in _ANNOTATION_FALLBACK_DIRS:
            candidates.append(os.path.join(d, annotation_file))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Khong tim thay annotation file '{annotation_file}' (da thu: {candidates})"
    )


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_transforms,
        pipe,
        task_prompt,
        data_root,
        prob=1.0,
        is_validation=False,
        annotation_file=None,
        min_area=MIN_ANN_AREA,
        max_objects=MAX_OBJECTS_PER_SAMPLE,
        min_mask_pixels=MIN_MASK_PIXELS,
        max_mask_attempts=MAX_MASK_ATTEMPTS,
        max_mask_ratio=MAX_MASK_RATIO,
        cross_mask_scale=CROSS_MASK_SCALE,
        cross_mask_rotation=CROSS_MASK_ROTATION,
        cross_mask_translation=CROSS_MASK_TRANSLATION,
        max_cross_mask_tries=MAX_CROSS_MASK_TRIES,
        **kwargs
    ):
        self.data_root = data_root
        self.is_validation = is_validation
        self.min_area = min_area
        self.max_objects = max_objects
        self.min_mask_pixels = min_mask_pixels
        self.max_mask_attempts = max_mask_attempts
        self.max_mask_ratio = max_mask_ratio
        self.cross_mask_scale = cross_mask_scale
        self.cross_mask_rotation = cross_mask_rotation
        self.cross_mask_translation = cross_mask_translation
        self.max_cross_mask_tries = max_cross_mask_tries

        with open(os.path.join(data_root, "metadata_run.json"), "r") as f:
            self.data = json.load(f)

        # Annotation file mặc định theo split; có thể được ghi đè qua config/metadata.
        if annotation_file is None:
            annotation_file = self.data[0].get("annotation_file", "instances_train2017.json")
        self.annotation_path = _resolve_annotation_path(data_root, annotation_file)

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
            elif isinstance(t, transforms.Resize):
                # Mask phải dùng NEAREST để giữ mask nhị phân (0/255), không tạo
                # pixel xám ở biên do nội suy (validation dùng Resize + CenterCrop).
                mask_transforms_list.append(transforms.Resize(
                    t.size, interpolation=transforms.InterpolationMode.NEAREST
                ))
            elif isinstance(t, (transforms.CenterCrop, transforms.RandomCrop)):
                mask_transforms_list.append(t)
            else:
                break

        if not mask_transforms_list:
            mask_transforms_list = [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(512),
            ]

        self.mask_transform = transforms.Compose(mask_transforms_list)

        # Load annotation COCO một lần, giữ lại annId của từng image (sau lọc).
        self.coco = _silent_coco(self.annotation_path)
        self._ann_ids_by_image = self._load_filtered_ann_ids()

    def _load_filtered_ann_ids(self):
        """Lọc annotation: bỏ iscrowd=1 và area quá nhỏ, giữ lại danh sách annId theo image."""
        ann_ids_by_image = {}
        img_ids = sorted(self.coco.getImgIds())
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if not ann_ids:
                continue
            anns = self.coco.loadAnns(ann_ids)
            valid = [
                ann["id"]
                for ann in anns
                if ann.get("iscrowd", 0) == 0 and ann.get("area", 0) >= self.min_area
            ]
            if valid:
                ann_ids_by_image[img_id] = valid
        return ann_ids_by_image

    def __len__(self):
        return len(self.data)

    def _random_mask(self, image_id, image_size, allow_other=True, rng=None):
        """Random chọn một subset (>=1) object hợp lệ của ảnh rồi hợp mask thành 1.

        Nếu ảnh không có annotation hợp lệ và allow_other=True, thử một ảnh bất kỳ
        khác trong dataset có annotation (phòng metadata lệch annotation).
        """
        if rng is None:
            rng = random
        ann_ids = self._ann_ids_by_image.get(image_id)

        if (not ann_ids) and allow_other:
            valid_ids = [iid for iid, anns in self._ann_ids_by_image.items() if anns]
            if valid_ids:
                image_id = rng.choice(valid_ids)
                ann_ids = self._ann_ids_by_image[image_id]

        if not ann_ids:
            # Không còn annotation nào khả dụng: mask đen (mô hình xem như không có gì để inpaint).
            # Image.new nhận size dạng (W, H) trong khi image_size là (H, W).
            return Image.new("L", (image_size[1], image_size[0]), 0)

        # Random chọn 1..max(1, max_objects) object trong ảnh, không vượt quá số ann có sẵn.
        max_k = min(self.max_objects, len(ann_ids))
        k = rng.randint(1, max_k)
        chosen = rng.sample(ann_ids, k)

        mask = np.zeros(image_size, dtype=np.uint8)
        for ann_id in chosen:
            ann_mask = self.coco.annToMask(self.coco.loadAnns([ann_id])[0])
            # annToMask trả về mask ở kích thước ảnh gốc; resize về image_size nếu khác.
            if ann_mask.shape != image_size:
                # PIL resize nhận (W, H); image_size là (H, W) nên phải đảo thứ tự.
                ann_mask = np.asarray(
                    Image.fromarray(ann_mask).resize(
                        (image_size[1], image_size[0]), Image.NEAREST
                    )
                )
            mask = np.maximum(mask, ann_mask)

        return Image.fromarray(np.uint8(mask))

    def _random_cross_image_mask(self, current_image_id, target_size, rng=None):
        """Object removal: lấy mask từ 1 object của ẢNH KHÁC, random affine rồi resize.

        Quy trình:
          1. Random chọn image_id khác (không trùng current_image_id).
          2. Lọc annotation hợp lệ (iscrowd=0, area đủ lớn).
          3. Random chọn 1 object.
          4. annToMask() -> mask ở kích thước ảnh B.
          5. Quy đổi mask về không gian ảnh A (contain, giữ aspect ratio).
          6. Random scale (CROSS_MASK_SCALE) + rotate + translate (affine).
          7. Paste lên canvas ảnh A.
        """
        if rng is None:
            rng = random
        width_A, height_A = target_size

        valid_ids = [iid for iid, anns in self._ann_ids_by_image.items() if anns]
        candidates = [iid for iid in valid_ids if iid != current_image_id]
        if not candidates:
            # Không có ảnh khác: fallback về chính ảnh này (mask từ ảnh A).
            return self._random_mask(current_image_id, (height_A, width_A), rng=rng)

        for _ in range(self.max_cross_mask_tries):
            other_id = rng.choice(candidates)
            ann_ids = self._ann_ids_by_image[other_id]
            ann_id = rng.choice(ann_ids)
            ann = self.coco.loadAnns([ann_id])[0]
            ann_mask = self.coco.annToMask(ann)  # kích thước ảnh B (H_B, W_B)

            if ann_mask.shape[1] == 0 or ann_mask.shape[0] == 0:
                continue

            mask_img = Image.fromarray(np.uint8(ann_mask * 255))  # L mode

            # 5. Crop về bounding box của object B để scale có ý nghĩa tương đối
            #    với ảnh A: scale=1.0 -> object phủ ~100% ảnh A, scale=0.3 -> ~30%.
            ys, xs = np.where(np.asarray(mask_img) > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            mask_img = mask_img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

            # 6. Affine: scale tương đối với ảnh A -> rotate -> translate
            scale = rng.uniform(*self.cross_mask_scale)
            angle = rng.uniform(*self.cross_mask_rotation)

            # scale bbox object B vào khung (scale * ảnh A), giữ aspect ratio.
            target_w = max(1, int(width_A * scale))
            target_h = max(1, int(height_A * scale))
            fit = min(target_w / bw, target_h / bh)
            new_w = max(1, int(bw * fit))
            new_h = max(1, int(bh * fit))
            mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)

            # rotate (expand để không cắt mask)
            mask_img = mask_img.rotate(angle, resample=Image.NEAREST, expand=True)

            # translate ngẫu nhiên trong phạm vi cho phép
            max_dx = int(self.cross_mask_translation * width_A)
            max_dy = int(self.cross_mask_translation * height_A)
            dx = rng.randint(-max_dx, max_dx)
            dy = rng.randint(-max_dy, max_dy)

            # 7. Paste lên canvas ảnh A
            canvas = Image.new("L", (width_A, height_A), 0)
            canvas.paste(mask_img, (dx, dy))
            mask_img = canvas

            mask_np = np.asarray(mask_img)
            ratio = float(np.count_nonzero(mask_np)) / (width_A * height_A)
            if self.max_mask_ratio > 0 and ratio > self.max_mask_ratio:
                # Mask quá to sau affine -> thử ảnh/object khác.
                continue
            return mask_img

        # Sau nhiều lần thử vẫn không ổn: fallback mask ảnh hiện tại.
        return self._random_mask(current_image_id, (height_A, width_A), rng=rng)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        # Keep validation loss reproducible: training still samples the task at
        # random, while validation assigns each sample to one fixed task.
        # (Phải quyết định task TRƯỚC vì cách sinh mask phụ thuộc task.)
        # Validation dùng RNG cục bộ seeded theo idx: mask, affine, seed transform
        # và tradeoff đều tái lập được giữa các lần đánh giá. Training giữ random toàn cục.
        if self.is_validation:
            rng = random.Random(idx)
        else:
            rng = random

        task_key = (
            "text_guided_object_synthesis"
            if (self.is_validation and idx % 2 == 0) or (not self.is_validation and rng.random() < 0.5)
            else "object_removal"
        )

        image_size = (image.size[1], image.size[0])  # (H, W)

        # Sinh mask theo task:
        #   - text-guided: mask của chính ảnh (object của ảnh A).
        #   - object-removal: mask lấy từ ảnh KHÁC (Image B) + affine.
        # Kiểm tra mask thực tế model nhìn thấy (sau RandomCrop/Resize/Flip):
        # quá nhỏ (< min) hoặc quá to (> max_ratio) -> thử lại subset/ảnh khác.
        seed = rng.randint(0, 2**32)
        for _ in range(max(1, self.max_mask_attempts)):
            if task_key == "text_guided_object_synthesis":
                mask = self._random_mask(item.get("image_id"), image_size, rng=rng)
            else:
                mask = self._random_cross_image_mask(item.get("image_id"), (image.size[0], image.size[1]), rng=rng)

            torch.manual_seed(seed)
            random.seed(seed)
            mask_transformed = self.mask_transform(mask)

            mask_np = np.array(mask_transformed)
            mask_pixels = np.count_nonzero(mask_np)
            ratio = float(mask_pixels) / mask_np.size
            if mask_pixels >= self.min_mask_pixels and ratio <= self.max_mask_ratio:
                break
            # Mask không đạt yêu cầu: thử subset/ảnh khác với seed mới.
            seed = rng.randint(0, 2**32)

        # Transform image với seed cuối cùng (khớp crop/flip với mask đã chọn).
        torch.manual_seed(seed)
        random.seed(seed)
        pixel_values = self.image_transform(image)

        mask_tensor = torch.from_numpy(
            np.array(mask_transformed) / 255.0
        ).unsqueeze(0).float()

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

        tradeoff_weight = rng.uniform(0.5, 1.0)

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
