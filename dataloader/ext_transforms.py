import collections
from site import execsitecustomize
import torchvision
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision import transforms
import random 
import numbers
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from math import ceil

#
#  Extended Transforms for Semantic Segmentation
#
class TestTimeAugmentation:
    r'''
    Usage:
        tta = TestTimeAugmentation()
        image = Image.open('path/to/your/image.jpg')
        augmented_images = tta(image)
    '''

    def __init__(self):
        self.rescale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

    def __call__(self, image):
        original_width, original_height = image.size
        augmented_images = []

        for flip in [False, True]:
            for factor in self.rescale_factors:
                transform = transforms.Compose([
                    transforms.Resize((int(factor * original_height), int(factor * original_width))),
                    transforms.RandomHorizontalFlip(p=1.0) if flip else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

                augmented_image = transform(image)
                augmented_images.append(augmented_image)

        return augmented_images

class ExtColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0):
        super().__init__()
        self.colorjitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = p

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image.
            lbls (PIL Images): List of Labels.
        Returns:
            PIL Image: Color jittered image.
            PIL Image: Labels.
        """
        if self.prob < torch.rand(1):
            return img, lbls
        else:
            return self.colorjitter(img), lbls

    def __repr__(self):
        return self.__class__.__name__

class ExtRandomGrayscale(object):
    def __init__(self, p=0):
        super().__init__()
        self.gray = T.RandomGrayscale(p=p)

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image.
            lbls (PIL Images): List of Labels.
        Returns:
            PIL Image: Gray scaled image.
            PIL Image: Labels.
        """
        return self.gray(img), lbls

    def __repr__(self):
        return self.__class__.__name__

class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), [F.hflip(lbl) for lbl in lbls]
        return img, lbls

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbls):
        for t in self.transforms:
            img, lbls = t(img, lbls)
        return img, lbls

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size), [F.center_crop(lbl, self.size) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=InterpolationMode.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return F.resize(img, target_size, self.interpolation), \
                    [F.resize(lbl, target_size, InterpolationMode.NEAREST) for lbl in lbls]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.
    Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.
    Returns:
    A random scale value selected between minimum and maximum value.
    Raises:
    ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                    minval=min_scale_factor,
                                    maxval=max_scale_factor)

  # When step_size != 0, we randomly select one discrete value from [min, max].
    # num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    # scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    # shuffled_scale_factors = tf.random_shuffle(scale_factors)

    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = torch.linspace(min_scale_factor, max_scale_factor, num_steps)
    scale_factors=scale_factors[torch.randperm(scale_factors.size()[0])]
    
    return shuffled_scale_factors[0]

class ExtScale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
        return F.resize(img, target_size, self.interpolation), \
                [F.resize(lbl, target_size, Image.NEAREST) for lbl in lbls]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ExtRandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, lbls):
        """
            img (PIL Image): Image to be rotated.
            lbl (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center), \
                [F.rotate(lbl, angle, self.resample, self.expand, self.center) for lbl in lbls]

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), [F.hflip(lbl) for lbl in lbls]
        return img, lbls

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            return F.vflip(img), [F.vflip(lbl) for lbl in lbls]
        return img, lbls

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbls):
        h, w = img.size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0
        im = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        lbls = [F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2)) for lbl in lbls]
        return im, lbls

class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, dtype_list=['int']):
    # def __init__(self, normalize=True, dtype_list=['uint8'], dtype_list='uint8'):
        self.normalize = normalize
        self.dtype_list = dtype_list

    def __call__(self, pic, lbls):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbls list of (PIL Image or numpy.ndarray): Label to be converted to tensor (label and superpixel).
        Returns:
            Tensor: Converted image and label
        """
        assert(len(lbls) == len(self.dtype_list))

        if self.normalize:
            return F.to_tensor(pic), [torch.from_numpy(np.array(lbl, dtype=type)) for lbl, type in zip(lbls, self.dtype_list)]
        else:
            return torch.from_numpy(np.array(pic, dtype=np.float32).transpose(2, 0, 1)), \
                    [torch.from_numpy(np.array(lbl, dtype=type)) for lbl, type in zip(lbls, self.dtype_list)]

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbls):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(tensor, self.mean, self.std), lbls

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, pad_values=[255, 2048], padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size ### w, h
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_values = pad_values

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped. (W, H)
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size ### for pil image
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def pad_data(self, img, lbls):
        w, h = img.size
        th, tw = self.size

        if self.pad_if_needed:
            assert(len(self.pad_values) == len(lbls))
        # pad the height if needed
        if self.pad_if_needed and h < th: ### 512 < 512
            gap = ceil((th - h) / 2)
            img = F.pad(img, padding=(0, gap, 0, gap), fill=self.padding) ### left, top, right, bottom
            lbls = [F.pad(lbl, padding=(0, gap, 0, gap), fill=pad_value) for (lbl, pad_value) in zip(lbls, self.pad_values)]
            assert(len(lbls) == len(self.pad_values))

        # pad the width if needed
        if self.pad_if_needed and w < tw:
            gap = ceil((tw - w) / 2)
            img = F.pad(img, padding=(gap, 0, gap, 0), fill=self.padding) ### left, top, right, bottom
            lbls = [F.pad(lbl, padding=(gap, 0, gap, 0), fill=pad_value) for (lbl, pad_value) in zip(lbls, self.pad_values)]
            assert(len(lbls) == len(self.pad_values))

        return img, lbls

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        img, lbls = self.pad_data(img, lbls)
        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), [F.crop(lbl, i, j, h, w) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class ExtRandomCropMulti(ExtRandomCrop):
    """RandomCrop for multi-hot labeling"""

    def __init__(self, size, lbl_pad_value=None, sp_pad_values=None, padding=0, pad_if_needed=False):
        super().__init__(size, lbl_pad_value, sp_pad_values[0], padding, pad_if_needed)
        ''' pad value list '''
        self.pad_values = sp_pad_values
        self.pad_values.insert(0, self.lbl_pad_value)

    def pad_data(self, img, lbls):
        """
        Pad lbls as a list, where first element is multi-hot annotation and rests are superpixels
        """
        assert(len(lbls) == len(self.pad_values))
        w, h = img.size
        th, tw = self.size

        # pad the height if needed
        if self.pad_if_needed and h < th: ### 512 < 768
            gap = ceil((th - h) / 2)
            img = F.pad(img, padding=(0, gap, 0, gap), fill=self.padding) ### left, top, right, bottom
            lbls = [F.pad(lbl, padding=(0, gap, 0, gap), fill=self.pad_values[ldx]) for ldx, lbl in enumerate(lbls)]                   

        # pad the width if needed
        if self.pad_if_needed and img.size[1] < self.size[1]:
            gap = ceil((tw - w) / 2)
            img = F.pad(img, padding=(gap, 0, gap, 0), fill=self.padding) ### left, top, right, bottom
            lbls = [F.pad(lbl, padding=(gap, 0, gap, 0), fill=self.pad_values[ldx]) for ldx, lbl in enumerate(lbls)]

        return img, lbls

class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbls):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation), \
                [F.resize(lbl, self.size, InterpolationMode.NEAREST) for lbl in lbls]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 
    

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# class MultiScaleFlipAug(object):

#     def __init__(self, img_scale, img_ratio=None, flip=False, flip_direction='horizontal'):
#         if img_ratios is not None:
#             img_ratios = img_ratios if isinstance(img_ratios, list) else [img_ratios]
#         if img_scale is None:
#             self.img_scale = None
#         elif isinstance(img_scale, tuple):
#             assert len(img_scale) == 2
#             self.img_scale = [(int(img_scale[0] * ratio),
#                                int(img_scale[1] * ratio))
#                               for ratio in img_ratios]
#         else:
#             # mode 3: given multiple scales
#             self.img_scale = img_scale if isinstance(img_scale,
#                                                      list) else [img_scale]
#         self.flip = flip
#         self.img_ratios = img_ratios
#         self.flip_direction = flip_direction if isinstance(
#             flip_direction, list) else [flip_direction]
        
#     def __call__(self, img, lbls):
        
#         aug_data = []
#         if self.img_scale is None:
#             h, w = img.shape[:2]
#             img_scale = [(int(w * ratio), int(h * ratio))
#                          for ratio in self.img_ratios]
#         else:
#             img_scale = self.img_scale
#         # flip_aug = [False, True] if self.flip else [False]
#         for scale in img_scale:
#             img_half, lbls_half = ExtScale(0.5)(img, lbls)
#             img_75, lbls_75 = ExtScale(0.75)(img, lbls)

                    
#         # if random.random() < self.p:
#         #     return F.hflip(img), [F.hflip(lbl) for lbl in lbls]
#         return img, lbls

#     def __repr__(self):
#         return self.__class__.__name__ + '(p={})'.format(self.p)