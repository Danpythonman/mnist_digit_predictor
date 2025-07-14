from torch import Tensor


def tensor2img(image: Tensor):
    img = image / 2 + 0.5              # Un-normalize
    npimg = img.numpy()                # Tensor -> NumPy
    npimg = npimg.transpose((1, 2, 0)) # CxHxW -> HxWxC
    return npimg
