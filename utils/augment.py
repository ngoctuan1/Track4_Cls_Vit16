import torch
import numpy as np
import cv2

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
