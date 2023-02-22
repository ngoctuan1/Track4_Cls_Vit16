import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance

class Mixup(object):
    def __init__(self, alpha = 1, device="cpu"):
        self.alpha = alpha
        self.device = device

    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_inputs = lam * images + (1-lam)*images[index: ]
        labels_a, labels_b = labels, labels[index]
        return mixed_inputs, (labels_a, labels_b,lam)
    
class CutMix(object):
    def __init__(self, alpha = 1, device = "cpu"):
        self.alpha = alpha
        self.device = device

    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_inputs = images.clone()

        r = np.random.rand(1)
        w = int(images.size()[-2] * np.sqrt(1 - np.power(r, 2)))
        h = int(images.size()[-1] * np.sqrt(1 - np.power(r, 2)))
        x = np.random.randint(0, images.size()[-2] - w)
        y = np.random.randint(0, images.size()[-1] - h)
        
        mixed_inputs[:, :, x:x+w, y:y+h] = images[index, :, x:x+w, y:y+h]
        labels_a, labels_b = labels, labels[index]
        return mixed_inputs, (labels_a, labels_b,lam)

class MixCut(object):
    def __init__(self, alpha=1.0, use_cuda=True):
        self.alpha = alpha
        self.use_cuda = use_cuda

    def __call__(self, inputs, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        if lam < 0.5:
            # Mixup
            batch_size = inputs.size()[0]
            if self.use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
            labels_a, labels_b = labels, labels[index]
        else:
            # Cutmix
            mixed_inputs = inputs.clone()
            batch_size = inputs.size()[0]
            if self.use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            r = np.random.rand(1)
            w = int(inputs.size()[-2] * np.sqrt(1 - np.power(r, 2)))
            h = int(inputs.size()[-1] * np.sqrt(1 - np.power(r, 2)))
            x = np.random.randint(0, inputs.size()[-2] - w)
            y = np.random.randint(0, inputs.size()[-1] - h)

            mixed_inputs[:, :, x:x+w, y:y+h] = inputs[index, :, x:x+w, y:y+h]
            labels_a, labels_b = labels, labels[index]
        
        return mixed_inputs, (labels_a, labels_b, lam)

class MSRCP(object):
    def __init__(self, sigmas = [12, 80, 250], s1 = 0.01, s2=0.01):
        self.sigmas = sigmas
        self.s1 = s1
        self.s2 = s2
        self.eps=np.finfo(np.double).eps
    def __call__(self, img):
        Int=np.sum(img,axis=2)/3
        Diffs=[]
        for sigma in self.sigmas:
            Diffs.append(np.log(Int+1)-np.log(self.gauss_blur(Int,sigma)+1))
        MSR=sum(Diffs)/3
        Int1=self.simplest_color_balance(MSR,self.s1,self.s2)
        B=np.max(img,axis=2)
        A=np.min(np.stack((255/(B+self.eps),Int1/(Int+self.eps)),axis=2),axis=-1)
        return (A[...,None]*img).astype('uint8')

    def gauss_blur_original(self,img,sigma):
        '''suitable for 1 or 3 channel image'''
        row_filter=self.get_gauss_kernel(sigma,1)
        t=cv2.filter2D(img,-1,row_filter[...,None])
        return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

    def gauss_blur_recursive(img,sigma):
        '''refer to “Recursive implementation of the Gaussian filter”
        (doi: 10.1016/0165-1684(95)00020-E). Paper considers it faster than 
        FFT(Fast Fourier Transform) implementation of a Gaussian filter. 
        Suitable for 1 or 3 channel image'''
        pass

    def gauss_blur(self,img,sigma,method='original'):
        if method=='original':
            return self.gauss_blur_original(img,sigma)
        elif method=='recursive':
            return self.gauss_blur_recursive(img,sigma)

    def simplest_color_balance(self,img_msrcr,s1,s2):
        '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb). 
        Only suitable for 1-channel image'''
        sort_img=np.sort(img_msrcr,None)
        N=img_msrcr.size
        Vmin=sort_img[int(N*s1)]
        Vmax=sort_img[int(N*(1-s2))-1]
        img_msrcr[img_msrcr<Vmin]=Vmin
        img_msrcr[img_msrcr>Vmax]=Vmax
        return (img_msrcr-Vmin)*255/(Vmax-Vmin)

    def get_gauss_kernel(self,sigma,dim=2):
        '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after 
        normalizing the 1D kernel, we can get 2D kernel version by 
        matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that 
        if you want to blur one image with a 2-D gaussian filter, you should separate 
        it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column 
        filter, one row filter): 1) blur image with first column filter, 2) blur the 
        result image of 1) with the second row filter. Analyse the time complexity: if 
        m&n is the shape of image, p&q is the size of 2-D filter, bluring image with 
        2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
        ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to 
        #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
        k_1D=np.arange(ksize)-ksize//2
        k_1D=np.exp(-k_1D**2/(2*sigma**2))
        k_1D=k_1D/np.sum(k_1D)
        if dim==1:
            return k_1D
        elif dim==2:
            return k_1D[:,None].dot(k_1D.reshape(1,-1))
def bgr2gray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gray2bgr(img):
  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def adjust_color(img, alpha=1, beta=None, gamma=0, backend='cv2'):
    r"""It blends the source image and its gray image:

    .. math::
        output = img * alpha + gray\_img * beta + gamma

    Args:
        img (ndarray): The input source image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.
        backend (str | None): The image processing backend type. Options are
            `cv2`, `pillow`, `None`. If backend is None, the global
            ``imread_backend`` specified by ``mmcv.use_backend()`` will be
            used. Defaults to None.

    Returns:
        ndarray: Colored image which has the same size and dtype as input.
    """
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        # Image.fromarray defaultly supports RGB, not BGR.
        pil_image = Image.fromarray(img[..., ::-1], mode='RGB')
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(alpha)
        return np.array(pil_image, dtype=img.dtype)[..., ::-1]
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.tile(gray_img[..., None], [1, 1, 3])
        if beta is None:
            beta = 1 - alpha
        colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
        if not colored_img.dtype == np.uint8:
            # Note when the dtype of `img` is not the default `np.uint8`
            # (e.g. np.float32), the value in `colored_img` got from cv2
            # is not guaranteed to be in range [0, 255], so here clip
            # is needed.
            colored_img = np.clip(colored_img, 0, 255)
        return colored_img.astype(img.dtype)


def adjust_brightness(img, factor=1., backend='cv2'):
    """Adjust image brightness.

    This function controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:

    .. math::
        output = img * factor + degenerated * (1 - factor)

    Args:
        img (ndarray): Image to be brightened.
        factor (float): A value controls the enhancement.
            Factor 1.0 returns the original image, lower
            factors mean less color (brightness, contrast,
            etc), and higher values more. Default 1.
        backend (str | None): The image processing backend type. Options are
            `cv2`, `pillow`, `None`. If backend is None, the global
            ``imread_backend`` specified by ``mmcv.use_backend()`` will be
            used. Defaults to None.

    Returns:
        ndarray: The brightened image.
    """
    if backend is None:
        backend = imread_backend
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        # Image.fromarray defaultly supports RGB, not BGR.
        pil_image = Image.fromarray(img[..., ::-1], mode='RGB')
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(factor)
        return np.array(pil_image, dtype=img.dtype)[..., ::-1]
    else:
        degenerated = np.zeros_like(img)
        # Note manually convert the dtype to np.float32, to
        # achieve as close results as PIL.ImageEnhance.Brightness.
        # Set beta=1-factor, and gamma=0
        brightened_img = cv2.addWeighted(
            img.astype(np.float32), factor, degenerated.astype(np.float32),
            1 - factor, 0)
        brightened_img = np.clip(brightened_img, 0, 255)
        return brightened_img.astype(img.dtype)


def auto_contrast(img, cutoff=0):
    """Auto adjust image contrast.

    This function maximize (normalize) image contrast by first removing cutoff
    percent of the lightest and darkest pixels from the histogram and remapping
    the image so that the darkest pixel becomes black (0), and the lightest
    becomes white (255).

    Args:
        img (ndarray): Image to be contrasted. BGR order.
        cutoff (int | float | tuple): The cutoff percent of the lightest and
            darkest pixels to be removed. If given as tuple, it shall be
            (low, high). Otherwise, the single value will be used for both.
            Defaults to 0.

    Returns:
        ndarray: The contrasted image.
    """

    def _auto_contrast_channel(im, c, cutoff):
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # Remove cut-off percent pixels from histo
        histo_sum = np.cumsum(histo)
        cut_low = histo_sum[-1] * cutoff[0] // 100
        cut_high = histo_sum[-1] - histo_sum[-1] * cutoff[1] // 100
        histo_sum = np.clip(histo_sum, cut_low, cut_high) - cut_low
        histo = np.concatenate([[histo_sum[0]], np.diff(histo_sum)], 0)

        # Compute mapping
        low, high = np.nonzero(histo)[0][0], np.nonzero(histo)[0][-1]
        # If all the values have been cut off, return the origin img
        if low >= high:
            return im
        scale = 255.0 / (high - low)
        offset = -low * scale
        lut = np.array(range(256))
        lut = lut * scale + offset
        lut = np.clip(lut, 0, 255)
        return lut[im]

    if isinstance(cutoff, (int, float)):
        cutoff = (cutoff, cutoff)
    else:
        assert isinstance(cutoff, tuple), 'cutoff must be of type int, ' \
            f'float or tuple, but got {type(cutoff)} instead.'
    # Auto adjusts contrast for each channel independently and then stacks
    # the result.
    s1 = _auto_contrast_channel(img, 0, cutoff)
    s2 = _auto_contrast_channel(img, 1, cutoff)
    s3 = _auto_contrast_channel(img, 2, cutoff)
    contrasted_img = np.stack([s1, s2, s3], axis=-1)
    return contrasted_img.astype(img.dtype)

