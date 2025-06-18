from skimage.exposure import equalize_adapthist
from numpy import stack, arange, array, save

def perform_adapthist_equalisation(image, nbins=256):
    red_corrected = equalize_adapthist(image[:,:,0], nbins=nbins)
    blue_corrected = equalize_adapthist(image[:,:,2], nbins=nbins)

    return stack([red_corrected, blue_corrected, blue_corrected], axis=2)

def get_mask_levels(masks, start:int=1):
    return array([l for l in arange(start, masks.max()+1) if (masks == l).any()])

def get_generations(levels):
    return array([l // 1000 for l in levels])

def decompose_masks(masks, reduce:bool=True, flexibility=5, return_levels:bool=False):
    levels = get_mask_levels(masks)
    result = []
    for l in levels:
        if reduce:
            result.append(remove_borders(masks == l, flexibility=flexibility))
        else:
            result.append(masks == l)

    if return_levels:
        return result, levels
    return result

def measure_coverage(nucleus, cell):
    return (nucleus * cell).sum() / nucleus.sum()

def save_to(cells, nuclei, fname):
    save(fname, stack([cells, nuclei], axis=2))