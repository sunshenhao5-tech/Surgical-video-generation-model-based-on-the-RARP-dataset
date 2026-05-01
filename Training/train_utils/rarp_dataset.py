"""
RARPDataset – multimodal dataloader for RARP-45 style surgical video data.

Directory layout expected for each video clip:
  <video_root>/
    frames/           00001.png … (1-indexed, T frames)
    flow image/       00001.png … (1-indexed, T-1 frames; flow[i] = frame[i]→frame[i+1])
    depth image/      00001.png … (1-indexed, T frames; stored as RGB with equal channels)
    instance1/masks_overall/  00000.png … (0-indexed, T frames; mask[i] ↔ frame[i+1])

Returns per sample:
  pixel_values : [T, 3, H, W]  float32  [0, 1]
  flows        : [T-1, 2, H, W] float32  decoded (u, v) in pixels at target resolution
  depths       : [T, 1, H, W]  float32  [0, 1]
  masks        : [T, 1, H, W]  float32  {0, 1} binary
  video_name   : str
"""

import os
import math
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Flow HSV → (u, v) decoding
# ---------------------------------------------------------------------------
# The stored flow images are produced by the standard flow_to_color() HSV
# encoding:
#   Hue (0-179 in OpenCV, i.e. 0-360°/2) → direction
#   Saturation (0-255) → magnitude (normalised to max_flow)
#   Value = 255 (always fully bright)
# We reverse this to recover approximate (u, v) vectors.

def _decode_flow_hsv(flow_img_np: np.ndarray, max_flow: float = 20.0) -> np.ndarray:
    """
    flow_img_np: HxWx3 uint8 RGB image (flow visualisation)
    Returns: HxWx2 float32 array (u, v) in pixels, capped to ±max_flow
    """
    import cv2
    # Convert RGB→HSV (OpenCV HSV: H∈[0,179], S,V∈[0,255])
    bgr = cv2.cvtColor(flow_img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h = hsv[..., 0]   # [0, 179]
    s = hsv[..., 1]   # [0, 255]  → magnitude

    angle = h / 179.0 * 2.0 * math.pi   # radians
    magnitude = (s / 255.0) * max_flow

    u = magnitude * np.cos(angle)
    v = magnitude * np.sin(angle)
    flow = np.stack([u, v], axis=-1)    # HxWx2
    return flow.astype(np.float32)


def _resize_flow(flow_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    flow_np: HxWx2 float32
    Bilinear resize + rescale magnitude proportionally.
    """
    src_h, src_w = flow_np.shape[:2]
    if src_h == target_h and src_w == target_w:
        return flow_np

    import cv2
    u = cv2.resize(flow_np[..., 0], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(flow_np[..., 1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Rescale displacement magnitudes
    u = u * (target_w / src_w)
    v = v * (target_h / src_h)
    return np.stack([u, v], axis=-1)


def _letterbox_image(img_pil: Image.Image, target_w: int, target_h: int,
                     resample=Image.BILINEAR, fill: int = 0) -> Image.Image:
    """
    Scale img_pil to fit inside (target_w, target_h) while preserving aspect
    ratio, then pad with `fill` value to exactly (target_w, target_h).
    For RGB images fill=(0,0,0); for L images fill=0.
    """
    src_w, src_h = img_pil.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = img_pil.resize((new_w, new_h), resample)
    mode = img_pil.mode
    if mode == "RGB":
        canvas = Image.new("RGB", (target_w, target_h), (fill, fill, fill))
    else:
        canvas = Image.new(mode, (target_w, target_h), fill)
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas


def _letterbox_flow(flow_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Scale flow_np (HxWx2) to fit inside (target_h, target_w) preserving
    aspect ratio, rescale displacement vectors, then zero-pad to target size.
    """
    src_h, src_w = flow_np.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    import cv2
    u = cv2.resize(flow_np[..., 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR) * scale
    v = cv2.resize(flow_np[..., 1], (new_w, new_h), interpolation=cv2.INTER_LINEAR) * scale
    canvas = np.zeros((target_h, target_w, 2), dtype=np.float32)
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = np.stack([u, v], axis=-1)
    return canvas


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_DATA_ROOTS = [
    "knotting_videos",
    "needleGrasping_videos",
    "needlePuncture_videos",
]


def _is_valid_clip(clip_dir: Path) -> bool:
    """Sanity check: all four modality directories exist and frames/depths are non-empty.
    At least one of instance1 or instance2 must have non-empty masks_overall."""
    frames_dir = clip_dir / "frames"
    flow_dir   = clip_dir / "flow image"
    depth_dir  = clip_dir / "depth image"
    if not (frames_dir.is_dir() and flow_dir.is_dir() and depth_dir.is_dir()):
        return False
    if len(list(frames_dir.glob("*.png"))) < 2:
        return False
    # Accept if at least one instance has masks
    for inst in ("instance1", "instance2"):
        mask_dir = clip_dir / inst / "masks_overall"
        if mask_dir.is_dir() and len(list(mask_dir.glob("*.png"))) > 0:
            return True
    return False


def _sorted_pngs(directory: Path):
    return sorted(directory.glob("*.png"), key=lambda p: p.name)


class RARPDataset(Dataset):
    """
    Multimodal dataset for RARP surgical video clips.

    Args:
        data_root      : root directory containing knotting_videos/ etc.
        sample_n_frames: number of frames T to sample per clip (including first frame).
        sample_stride  : temporal stride between consecutive sampled frames.
        sample_size    : [H, W] spatial resolution for all modalities.
        max_flow       : assumed max optical flow magnitude (pixels) for HSV decoding.
        val_split      : fraction of clips reserved for validation.
        split          : "train" or "val".
        seed           : random seed for reproducible split.
        augment        : whether to apply random horizontal flip (train only).
    """

    def __init__(
        self,
        data_root: str,
        sample_n_frames: int = 21,
        sample_stride: int = 1,
        sample_size: list = None,
        max_flow: float = 20.0,
        val_split: float = 0.1,
        split: str = "train",
        seed: int = 42,
        augment: bool = True,
    ):
        if sample_size is None:
            sample_size = [256, 256]

        self.data_root = Path(data_root)
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.sample_h, self.sample_w = sample_size[0], sample_size[1]
        self.max_flow = max_flow
        self.augment = augment and (split == "train")

        # ------------------------------------------------------------------
        # Enumerate all valid clip directories
        # ------------------------------------------------------------------
        all_clips = []
        for subset in _DATA_ROOTS:
            subset_dir = self.data_root / subset
            if not subset_dir.is_dir():
                continue
            for clip_dir in sorted(subset_dir.iterdir()):
                if clip_dir.is_dir() and _is_valid_clip(clip_dir):
                    all_clips.append(clip_dir)

        if len(all_clips) == 0:
            raise RuntimeError(f"No valid clips found under {data_root}")

        # ------------------------------------------------------------------
        # Deterministic train/val split (by clip, not by frame)
        # ------------------------------------------------------------------
        rng = random.Random(seed)
        indices = list(range(len(all_clips)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_clips) * val_split))
        val_idx = set(indices[:n_val])

        if split == "val":
            self.clips = [all_clips[i] for i in indices[:n_val]]
        else:
            self.clips = [all_clips[i] for i in indices[n_val:]]

        print(f"[RARPDataset] split={split}  clips={len(self.clips)}  "
              f"(total={len(all_clips)}, val={n_val})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Returns HxWx3 uint8."""
        return np.array(Image.open(path).convert("RGB"))

    def _load_depth(self, path: Path) -> np.ndarray:
        """Returns HxW float32 [0,1]. Depth stored as RGB(equal channels)."""
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[..., 0]  # all channels identical – take R
        return img.astype(np.float32) / 255.0

    def _load_mask(self, path: Path) -> np.ndarray:
        """Returns HxW float32 {0,1}."""
        img = np.array(Image.open(path).convert("L"))
        return (img > 127).astype(np.float32)

    def _load_combined_mask(self, clip_dir: Path, mask_fi: int) -> np.ndarray:
        """
        Load and OR-combine masks_overall from instance1 and instance2.
        instance1 = left arm, instance2 = right arm; union = all instruments.
        Falls back gracefully if one instance is missing or empty.
        Returns HxW float32 {0,1}.
        """
        combined = None
        for inst in ("instance1", "instance2"):
            mask_files = _sorted_pngs(clip_dir / inst / "masks_overall")
            if len(mask_files) == 0:
                continue
            idx = min(mask_fi, len(mask_files) - 1)
            m = (np.array(Image.open(mask_files[idx]).convert("L")) > 127)
            combined = m if combined is None else (combined | m)
        if combined is None:
            # Should not happen after _is_valid_clip, but be safe
            return np.zeros((1, 1), dtype=np.float32)
        return combined.astype(np.float32)

    def _load_flow(self, path: Path) -> np.ndarray:
        """Returns HxWx2 float32 (u,v) pixels."""
        img = self._load_rgb(path)
        return _decode_flow_hsv(img, self.max_flow)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_frame_indices(self, n_available: int) -> list:
        """
        Sample `sample_n_frames` indices with stride `sample_stride` from
        [0, n_available). Falls back to stride=1 if the clip is too short.
        """
        needed = (self.sample_n_frames - 1) * self.sample_stride + 1
        if n_available >= needed:
            stride = self.sample_stride
        else:
            stride = max(1, (n_available - 1) // max(1, self.sample_n_frames - 1))

        max_start = n_available - (self.sample_n_frames - 1) * stride - 1
        max_start = max(0, max_start)
        start = random.randint(0, max_start)
        indices = [start + i * stride for i in range(self.sample_n_frames)]
        # Clamp to valid range
        indices = [min(i, n_available - 1) for i in indices]
        return indices

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def _load_clip(self, clip_dir: Path):
        frame_files = _sorted_pngs(clip_dir / "frames")
        flow_files  = _sorted_pngs(clip_dir / "flow image")
        depth_files = _sorted_pngs(clip_dir / "depth image")
        # mask files are loaded per-frame via _load_combined_mask (instance1 OR instance2)

        # Alignment:
        #   frame[i]  (1-indexed filename, 0-indexed list position)
        #   depth[i]  ↔ frame[i]
        #   mask[i]   ↔ frame[i+1]  (mask is 0-indexed, frame 1-indexed)
        #   flow[i]   ↔ frame[i]→frame[i+1]
        n_frames = len(frame_files)
        n_flows  = len(flow_files)   # == n_frames - 1

        frame_indices = self._sample_frame_indices(n_frames)

        # ------------------------------------------------------------------
        # Random horizontal flip (consistent across all modalities)
        # ------------------------------------------------------------------
        do_flip = self.augment and (random.random() < 0.5)

        rgb_list, depth_list, mask_list = [], [], []
        for fi in frame_indices:
            # RGB
            rgb = self._load_rgb(frame_files[fi])
            rgb_pil = _letterbox_image(Image.fromarray(rgb), self.sample_w, self.sample_h,
                                       resample=Image.BILINEAR, fill=0)
            if do_flip:
                rgb_pil = TF.hflip(rgb_pil)
            rgb_list.append(np.array(rgb_pil))

            # Depth
            d = self._load_depth(depth_files[fi])
            d_pil = _letterbox_image(Image.fromarray((d * 255).astype(np.uint8)),
                                     self.sample_w, self.sample_h,
                                     resample=Image.BILINEAR, fill=0)
            if do_flip:
                d_pil = TF.hflip(d_pil)
            depth_list.append(np.array(d_pil).astype(np.float32) / 255.0)

            # Mask: OR-combine instance1 + instance2
            # mask[fi] ↔ frame[fi+1]; for frame[fi] use mask[fi-1] (clamped to 0)
            mask_fi = max(0, fi - 1)
            m = self._load_combined_mask(clip_dir, mask_fi)
            m_pil = _letterbox_image(Image.fromarray((m * 255).astype(np.uint8)),
                                     self.sample_w, self.sample_h,
                                     resample=Image.NEAREST, fill=0)
            if do_flip:
                m_pil = TF.hflip(m_pil)
            mask_list.append(np.array(m_pil).astype(np.float32) / 255.0)

        # ------------------------------------------------------------------
        # Optical flow – T-1 flows between consecutive sampled frames
        # ------------------------------------------------------------------
        flow_list = []
        for k in range(len(frame_indices) - 1):
            fi = frame_indices[k]
            # flow[fi] represents motion from frame[fi] to frame[fi+1]
            # flow files are 1-indexed starting at 00001, list position == fi
            flow_fi = min(fi, n_flows - 1)
            fl = self._load_flow(flow_files[flow_fi])
            fl_resized = _letterbox_flow(fl, self.sample_h, self.sample_w)
            if do_flip:
                fl_resized = fl_resized[:, ::-1, :].copy()
                fl_resized[..., 0] *= -1   # u component flips sign
            flow_list.append(fl_resized)

        # ------------------------------------------------------------------
        # Convert to tensors
        # ------------------------------------------------------------------
        # RGB: [T,3,H,W] float32 [0,1]
        rgb_np = np.stack(rgb_list, axis=0).astype(np.float32) / 255.0
        pixel_values = torch.from_numpy(rgb_np).permute(0, 3, 1, 2)

        # Depth: [T,1,H,W] float32 [0,1]
        depth_np = np.stack(depth_list, axis=0)[:, np.newaxis]
        depths = torch.from_numpy(depth_np)

        # Mask: [T,1,H,W] float32 {0,1}
        mask_np = np.stack(mask_list, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(mask_np)

        # Flow: [T-1,2,H,W] float32
        flow_np = np.stack(flow_list, axis=0).transpose(0, 3, 1, 2)
        flows = torch.from_numpy(flow_np)

        return pixel_values, flows, depths, masks

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx % len(self.clips)]
        try:
            pixel_values, flows, depths, masks = self._load_clip(clip_dir)
        except Exception as e:
            # Fallback: return adjacent index to avoid crashing training
            print(f"[RARPDataset] WARNING: failed to load {clip_dir}: {e}")
            return self.__getitem__((idx + 1) % len(self.clips))

        return {
            "pixel_values": pixel_values,   # [T, 3, H, W]  [0,1]
            "flows":        flows,           # [T-1, 2, H, W]
            "depths":       depths,          # [T, 1, H, W]  [0,1]
            "masks":        masks,           # [T, 1, H, W]  {0,1}
            "video_name":   str(clip_dir.name),
        }
