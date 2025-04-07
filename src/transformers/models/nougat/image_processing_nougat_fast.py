# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for Nougat."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ImageInput,
    PILImageResampling,
    ChannelDimension,
    make_list_of_images,
    # valid_images,
    # SizeDict,
    # infer_channel_dimension_format,
    validate_preprocess_arguments,
    # is_scaled_image,
    ImageType,
    get_image_type,
    pil_torch_interpolation_mapping,
    # validate_kwargs
)
# from ...utils import add_start_docstrings
from ...image_processing_utils import get_size_dict
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    # infer_channel_dimension_format,
    # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...utils import (
    # add_start_docstrings,
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    # logging,
    filter_out_non_signature_kwargs,
    logging
)

# from ...image_transforms import (
#     # get_resize_output_image_size,
#     # pad,
#     # resize,
#     # to_channel_dimension_format,
#     # to_pil_image,
# )
from ...processing_utils import Unpack

from typing import Dict, List, Optional, Union
from PIL import Image
logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn.functional as F
if is_torchvision_available():
    import torchvision
class NougatFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]
    do_pad: Optional[bool]

@add_start_docstrings(
    "Constructs a fast Nougat image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class NougatImageProcessorFast(BaseImageProcessorFast):

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 896, "width": 672}
    default_to_square = None
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None
    valid_kwargs = NougatFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[NougatFastImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        size = size if size is not None else {"height": 896, "width": 672}

        if isinstance(size, (tuple, list)):
            size = size[::-1]
        self.size = get_size_dict(size)
        # Set other attributes from kwargs with defaults
        self.do_resize = kwargs.pop("do_resize", True)
        self.resample = kwargs.pop("resample", PILImageResampling.BILINEAR)
        self.do_rescale = kwargs.pop("do_rescale", True)
        self.do_normalize = kwargs.pop("do_normalize", True)
        self.image_mean = kwargs.pop("image_mean", IMAGENET_DEFAULT_MEAN)
        self.image_std = kwargs.pop("image_std", IMAGENET_DEFAULT_STD)
        self.default_to_square = kwargs.pop("default_to_square", None)
        self.do_center_crop = kwargs.pop("do_center_crop", None)
        self.crop_size = kwargs.pop("crop_size", None)
        self.do_convert_rgb = kwargs.pop("do_convert_rgb", None)

        # Call parent class initialization with remaining kwargs
        super().__init__(**kwargs)

    def python_find_non_zero(self, image: torch.tensor):
        non_zero_indices = torch.nonzero(image, as_tuple=False)
        idxvec = non_zero_indices[:, [1, 0]]  # swap columns (x, y)
        idxvec = idxvec.view(-1, 1, 2)  # reshape to (-1, 1, 2)
        return idxvec

    def infer_channel_dimension_format(
            self,
            image: torch.Tensor,
            num_channels: Optional[Union[int, tuple[int, ...]]] = None
    ) -> ChannelDimension:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, got {type(image)}")

        num_channels = (num_channels,) if isinstance(num_channels, int) else (num_channels or (1, 3, 4))

        if image.ndim == 3:
            first_dim, last_dim = 0, 2
        elif image.ndim == 4:
            first_dim, last_dim = 1, 3
        else:
            raise ValueError(f"Unsupported number of image dimensions: {image.ndim}. Expected 3D or 4D tensor.")

        shape = image.shape
        first_val = shape[first_dim]
        last_val = shape[last_dim]

        if first_val in num_channels and last_val in num_channels:
            logger.warning(
                f"The channel dimension is ambiguous. Got image shape {shape}. Assuming channels are the first dimension."
            )
            return ChannelDimension.FIRST
        elif first_val in num_channels:
            return ChannelDimension.FIRST
        elif last_val in num_channels:
            return ChannelDimension.LAST

        raise ValueError(
            f"Unable to infer channel dimension format from shape {shape} with num_channels={num_channels}"
        )

    def to_channel_dimension_format(
            self,
            image: torch.Tensor,
            channel_dim: Union[ChannelDimension, str],
            input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, got {type(image)}")

        channel_dim = ChannelDimension(channel_dim)
        if input_channel_dim is None:
            input_channel_dim = self.infer_channel_dimension_format(image)
        else:
            input_channel_dim = ChannelDimension(input_channel_dim)

        if input_channel_dim == channel_dim:
            return image

        # Handle 3D (single image)
        if image.ndim == 3:
            if input_channel_dim == ChannelDimension.FIRST and channel_dim == ChannelDimension.LAST:
                return image.permute(1, 2, 0)  # CHW -> HWC
            elif input_channel_dim == ChannelDimension.LAST and channel_dim == ChannelDimension.FIRST:
                return image.permute(2, 0, 1)  # HWC -> CHW

        # Handle 4D (batch of images)
        elif image.ndim == 4:
            if input_channel_dim == ChannelDimension.FIRST and channel_dim == ChannelDimension.LAST:
                return image.permute(0, 2, 3, 1)  # BCHW -> BHWC
            elif input_channel_dim == ChannelDimension.LAST and channel_dim == ChannelDimension.FIRST:
                return image.permute(0, 3, 1, 2)  # BHWC -> BCHW

        raise ValueError(
            f"Unsupported conversion from {input_channel_dim} to {channel_dim} for tensor with shape {image.shape}")

    def torch_bounding_rect(self, coordinates: torch.Tensor):

        if coordinates.dim() == 3:
            coordinates = coordinates.squeeze(1)

        min_values = torch.min(coordinates, dim=0).values.int()
        max_values = torch.max(coordinates, dim=0).values.int()

        x_min, y_min = min_values[0].item(), min_values[1].item()
        width = max_values[0].item() - x_min + 1
        height = max_values[1].item() - y_min + 1

        return x_min, y_min, width, height

    def convert_channel_format(
            self,
            tensor: torch.Tensor,
            input_format: Union[str, ChannelDimension],
            output_format: Union[str, ChannelDimension]
    ) -> torch.Tensor:

        # Normalize input formats to ChannelDimension if provided as strings
        input_format = ChannelDimension(input_format) if isinstance(input_format, str) else input_format
        output_format = ChannelDimension(output_format) if isinstance(output_format, str) else output_format

        # If the formats are the same, return the tensor as is
        if input_format == output_format:
            return tensor

        # Convert between "CHW" (Channel-Height-Width) and "HWC" (Height-Width-Channel)
        if input_format == ChannelDimension.FIRST and output_format == ChannelDimension.LAST:
            return tensor.permute(1, 2, 0)  # CHW -> HWC
        elif input_format == ChannelDimension.LAST and output_format == ChannelDimension.FIRST:
            return tensor.permute(2, 0, 1)  # HWC -> CHW
        else:
            raise ValueError(f"Unsupported format conversion: {input_format} -> {output_format}")


    def crop_margin(
            self,
            image: torch.Tensor,
            gray_threshold: int = 200,
            data_format: Union[str, ChannelDimension] = "CHW",  # options: "CHW" or "HWC"
    ) -> torch.Tensor:

        # Normalize data_format to ChannelDimension if provided as string
        data_format = ChannelDimension(data_format) if isinstance(data_format, str) else data_format

        # Convert to CHW for processing if needed
        if data_format == ChannelDimension.LAST:  # HWC -> CHW
            image = image.permute(2, 0, 1)

        # Convert to grayscale using standard luminance weights if 3-channel
        if image.size(0) == 3:
            gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            gray = image[0]  # Assume single channel

        # Normalize grayscale to 0-255
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-5) * 255
        gray_mask = gray < gray_threshold

        # Find non-gray (non-margin) indices
        nonzero_indices = torch.nonzero(~gray_mask)

        if nonzero_indices.numel() == 0:
            # Return the original image if no margin found
            return image.permute(1, 2, 0) if data_format == ChannelDimension.LAST else image

        # Get the min and max coordinates for cropping
        y_min = nonzero_indices[:, 0].min().item()
        y_max = nonzero_indices[:, 0].max().item()
        x_min = nonzero_indices[:, 1].min().item()
        x_max = nonzero_indices[:, 1].max().item()

        # Crop the image
        cropped = image[:, y_min: y_max + 1, x_min: x_max + 1]

        # Convert back to original data format if needed
        return cropped.permute(1, 2, 0) if data_format == ChannelDimension.LAST else cropped

    def align_long_axis(
            self,
            image: torch.Tensor,
            size: Dict[str, int],
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> torch.Tensor:

        # Normalize format inputs
        input_data_format = (
            ChannelDimension(input_data_format)
            if input_data_format is not None
            else self.infer_channel_dimension_format(image)
        )

        # Convert to HWC for easier manipulation
        image_hwc = self.to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

        # Get dimensions of input image
        input_height, input_width = image_hwc.shape[0], image_hwc.shape[1]
        output_height, output_width = size["height"], size["width"]

        # Determine if rotation is necessary
        rotate = (output_width < output_height and input_width > input_height) or \
                 (output_width > output_height and input_width < input_height)

        if rotate:
            # Rotate 90 degrees counter-clockwise (HWC)
            image_hwc = torch.rot90(image_hwc, k=1, dims=(0, 1))

        # Convert back to requested data format if needed
        if data_format is not None:
            image_hwc = self.to_channel_dimension_format(image_hwc, data_format, ChannelDimension.LAST)
        elif input_data_format == ChannelDimension.FIRST:
            image_hwc = self.to_channel_dimension_format(image_hwc, ChannelDimension.FIRST, ChannelDimension.LAST)

        return image_hwc

    def pad_image(
            self,
            image: torch.Tensor,
            size: Dict[str, int],
            data_format: Optional[Union[ChannelDimension, str]] = None,
            input_data_format: Optional[Union[ChannelDimension, str]] = None
    ) -> torch.Tensor:

        # Infer input data format if not provided
        input_data_format = (
            ChannelDimension(input_data_format)
            if input_data_format is not None
            else self.infer_channel_dimension_format(image)
        )

        # Convert to CHW for padding
        image_chw = self.to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        # Support both 3D (C, H, W) and 4D (B, C, H, W)
        if image_chw.ndim == 3:
            _, input_height, input_width = image_chw.shape
        elif image_chw.ndim == 4:
            _, _, input_height, input_width = image_chw.shape
        else:
            raise ValueError(f"Unsupported image dimensions: {image_chw.shape}")

        output_height, output_width = size["height"], size["width"]
        delta_height = output_height - input_height
        delta_width = output_width - input_width

        pad_top = delta_height // 2
        pad_bottom = delta_height - pad_top
        pad_left = delta_width // 2
        pad_right = delta_width - pad_left

        padding = [pad_left, pad_right, pad_top, pad_bottom]

        # F.pad expects 4D for batch, 3D for single image
        padded = (
            # F.pad(image_chw, padding, mode='constant', value=0)
            F.pad(image_chw, padding)
            if image_chw.ndim == 3
            # else F.pad(image_chw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            else F.pad(image_chw, [pad_left, pad_right, pad_top, pad_bottom])
        )

        # Convert back to desired format
        if data_format is not None:
            padded = self.to_channel_dimension_format(padded, data_format, ChannelDimension.FIRST)
        elif input_data_format == ChannelDimension.LAST:
            padded = self.to_channel_dimension_format(padded, ChannelDimension.LAST, ChannelDimension.FIRST)

        return padded

    def thumbnail(
            self,
            image: torch.Tensor,
            size: Dict[str, int],
            resample: Image.Resampling = Image.Resampling.BICUBIC,
            data_format: Optional[Union[ChannelDimension, str]] = None,
            input_data_format: Optional[Union[ChannelDimension, str]] = None,
    ) -> torch.Tensor:

        # Infer and normalize input format
        input_data_format = (
            ChannelDimension(input_data_format)
            if input_data_format is not None
            else self.infer_channel_dimension_format(image)
        )

        # Convert to CHW
        image_chw = self.to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        # Extract dimensions
        _, input_height, input_width = image_chw.shape
        max_height, max_width = size["height"], size["width"]

        # Compute aspect-ratio-preserving size
        scale = min(max_height / input_height, max_width / input_width, 1.0)
        new_height = int(input_height * scale)
        new_width = int(input_width * scale)

        if new_height == input_height and new_width == input_width:
            return image  # no resizing needed

        # Resize using F.interpolate (expects BCHW)
        image_resized = F.interpolate(
            image_chw.unsqueeze(0),
            size=(new_height, new_width),
            mode=resample.value,  # InterpolationMode.BICUBIC -> "bicubic"
            align_corners=False if resample in {Image.Resampling.BILINEAR, Image.Resampling.BICUBIC} else None
        ).squeeze(0)

        # Convert back to desired format
        if data_format is not None:
            image_resized = self.to_channel_dimension_format(image_resized, data_format, ChannelDimension.FIRST)
        elif input_data_format == ChannelDimension.LAST:
            image_resized = self.to_channel_dimension_format(image_resized, ChannelDimension.LAST, ChannelDimension.FIRST)

        return image_resized

    def resize(
            self,
            image: torch.Tensor,
            size: Dict[str, int],
            resample: Image.Resampling = Image.Resampling.BICUBIC,
            data_format: Optional[Union[ChannelDimension, str]] = None,
            input_data_format: Optional[Union[ChannelDimension, str]] = None,
    ) -> torch.Tensor:

        input_data_format = (
            ChannelDimension(input_data_format)
            if input_data_format is not None
            else self.infer_channel_dimension_format(image)
        )

        # Convert input to CHW
        image_chw = self.to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        # Apply resizing using interpolate
        resized = F.interpolate(
            image_chw.unsqueeze(0),  # Add batch dimension
            size=(size["height"], size["width"]),
            mode=resample.value,
            align_corners=False if resample in {Image.Resampling.BILINEAR, Image.Resampling.BICUBIC} else None
        ).squeeze(0)  # Remove batch dimension

        # Convert back if necessary
        if data_format is not None:
            resized = self.to_channel_dimension_format(resized, data_format, ChannelDimension.FIRST)
        elif input_data_format == ChannelDimension.LAST:
            resized = self.to_channel_dimension_format(resized, ChannelDimension.LAST, ChannelDimension.FIRST)

        return resized

    @filter_out_non_signature_kwargs()
    def preprocess(
            self,
            images: ImageInput,
            do_crop_margin: Optional[bool] = None,
            do_resize: Optional[bool] = None,
            size: Dict[str, int] = None,
            resample: PILImageResampling = None,
            do_thumbnail: Optional[bool] = None,
            do_align_long_axis: Optional[bool] = None,
            do_pad: Optional[bool] = None,
            do_rescale: Optional[bool] = None,
            rescale_factor: Union[int, float] = None,
            do_normalize: Optional[bool] = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            **kwargs
    ) -> BatchFeature:

        do_crop_margin = do_crop_margin if do_crop_margin is not None else self.do_crop_margin
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        device = kwargs.pop("device", None)

        images = make_list_of_images(images)


        images = make_list_of_images(images)
        image_type = get_image_type(images[0])

        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")
        # validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)


        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=size,  # There is no pad divisibility in this processor, but pad requires the size arg.
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        if image_type == ImageType.PIL:
            images = [torchvision.transforms.functional.pil_to_tensor(image) for image in images]
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            images = [torch.from_numpy(image).contiguous() for image in images]

        if device is not None:
            images = [image.to(device) for image in images]

        # We assume that all images have the same channel dimension format.
        if input_data_format is None:
            input_data_format = self.infer_channel_dimension_format(images[0])
        if input_data_format == ChannelDimension.LAST:
            images = [image.permute(2, 0, 1).contiguous() for image in images]
            input_data_format = ChannelDimension.FIRST

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = self.infer_channel_dimension_format(images[0])

        if do_crop_margin:
            images = [self.crop_margin(image, data_format=input_data_format) for image in images]

        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size, input_data_format=input_data_format) for image in images]

        resample = (
            pil_torch_interpolation_mapping[resample]
            if isinstance(resample, (PILImageResampling, int))
            else resample
        )

        if do_resize:

            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_thumbnail:
            images = [self.thumbnail(image=image, size=size, resample=resample) for image in images]

        if do_pad:
            images = [self.pad_image(image=image, size=size, input_data_format=input_data_format) for image in images]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            self.to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        pixel_tensor = torch.stack(images)
        data = {"pixel_values": pixel_tensor}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["NougatImageProcessorFast"]
