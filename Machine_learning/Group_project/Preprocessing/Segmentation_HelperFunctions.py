from numpy import copy, stack, arange, meshgrid, nan, ones, arange, stack
from matplotlib.pyplot import subplots, suptitle, tight_layout, savefig, show
from cellpose.models import Cellpose
from cellpose.io import imread
from General_HelperFunctions import *

def cell_nucleus_segmentation(image, cell_diam=100, nucleus_diam=80, threshold=0, min_coverage=0.99):
    
    # Predict the cell and nucleus masks:
    cells = Cellpose(model_type='cyto3', gpu=True).eval(image, channels=[3,1], diameter=cell_diam, cellprob_threshold=threshold)[0]
    nuclei = Cellpose(model_type='nuclei', gpu=True).eval(image, channels=[1,0], diameter=nucleus_diam, cellprob_threshold=threshold)[0]

    # Decompose from a 2D matrix to a list of 2D matrices, each with their own cell/nucleus:
    cell_masks = decompose_masks(cells, reduce=False)
    nucleus_masks = decompose_masks(nuclei, reduce=False)

    # Create a matrix, which stores the coverages between each cell and each nucleus:
    coverage_matrix = nan * ones(shape=(len(cell_masks), len(nucleus_masks)))
    for i, cell in enumerate(cell_masks):
        for j, nucleus in enumerate(nucleus_masks):
            coverage_matrix[i,j] = measure_coverage(nucleus, cell)

    # Interpret the matrix:
    # - If a cell completely covers a single nucleus, and doesn't collide with other nuclei, it passes the test
    unique_cells = []; unique_nuclei = []
    for i, cell in enumerate(cell_masks):
        column = coverage_matrix[i,:]

        if (column >= min_coverage).any():
            j = arange(len(column))[column == column.max()][0]
            
            unique_cells.append(i)
            unique_nuclei.append(j)

    # Combine into Cellpose-like outputs:
    unique_cell_masks = []
    unique_nucleus_masks = []
    count = 1
    for cell_idx, nucleus_idx in zip(unique_cells, unique_nuclei):
        unique_cell_masks.append(count * cell_masks[cell_idx])
        unique_nucleus_masks.append(count * nucleus_masks[nucleus_idx])

        count = int(count + 1)

    unique_cell_masks = stack(unique_cell_masks, axis=2).sum(axis=2)
    unique_nucleus_masks = stack(unique_nucleus_masks, axis=2).sum(axis=2)

    return unique_cell_masks, unique_nucleus_masks

def combined_segmentation(fname, cell_diams=[100, 100], nucleus_diams=[80, 80], threshold=0, show_plot=False, correction=perform_adapthist_equalisation, min_coverage=0.99, save_to=None):

    image = imread(fname)
    im = correction(copy(image))
    xx, yy = meshgrid(arange(image.shape[0]), arange(image.shape[1]))

    if show_plot:
        fig, axes = subplots(min([len(cell_diams), len(nucleus_diams)]), 2, figsize=(8,4*min([len(cell_diams), len(nucleus_diams)])), dpi=300)
        suptitle(fname.split('/')[-1])
    
    for i, (cell_diam, nucleus_diam) in enumerate(zip(cell_diams, nucleus_diams)):
        cells, nuclei = cell_nucleus_segmentation(im, cell_diam=cell_diam, nucleus_diam=nucleus_diam, threshold=threshold, min_coverage=min_coverage)
        print(f"{cells.max()} new cell/nucleus pairs")

        if show_plot:
            axes[i,0].imshow(im, origin='lower')
            axes[i,1].imshow(image, origin='lower')

        if i == 0:
            combined_cells = cells
            combined_nuclei = nuclei
        else:
            combined_cells[cells != 0] = cells[cells != 0] + int(i * 1e3)
            combined_nuclei[nuclei != 0] = nuclei[nuclei != 0] + int(i * 1e3)

        mask = (combined_cells != 0)
        channels = []
        for j in range(im.shape[2]):
            colour = im[:,:,j]
            colour[mask] = 0
            channels.append(colour)

        im = correction(stack(channels, axis=2))

        if show_plot:
            axes[i,0].contour(xx, yy, cells, colors='fuchsia', levels=get_mask_levels(combined_cells, start=0), linewidths=0.3)
            axes[i,0].contour(xx, yy, nuclei, colors='gold', levels=get_mask_levels(combined_nuclei, start=0), linewidths=0.3)
            axes[i,1].contour(xx, yy, combined_cells, colors='fuchsia', levels=get_mask_levels(combined_cells, start=0), linewidths=0.3)
            axes[i,1].contour(xx, yy, combined_nuclei, colors='gold', levels=get_mask_levels(combined_nuclei, start=0), linewidths=0.3)

            axes[i,0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            axes[i,1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

            axes[i,0].set_ylabel(f"{nuclei.max()} new masks")
            axes[i,0].set_title("Input image"); axes[i,1].set_title("Original image")

    if show_plot:
        tight_layout()
        if save_to is not None:
            savefig(save_to, bbox_inches='tight', dpi=300)
        show()

    return combined_cells, combined_nuclei