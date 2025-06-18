def nearest_neighbour(points):
    from numpy import meshgrid

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xg1, xg2 = meshgrid(xs, xs, indexing='ij')
    yg1, yg2 = meshgrid(ys, ys, indexing='ij')

    dx2, dy2 = (xg1 - xg2)**2, (yg1 - yg2)**2
    d = (dx2 + dy2)**0.5
    d[d==0] = 1e8

    return d.min(axis=1)

def gaussian_normalisation(arr):
    from numpy import nanmean, nanstd
    return (arr - nanmean(arr)) / nanstd(arr)

### Image Corrections

def ZtoRGB(channel):
    from numpy import uint8, where, isfinite, maximum, minimum
    y = maximum(minimum(channel, 5), -5)
    return uint8(where(isfinite(channel), 1 + (255-1) * (y - (-5)) / (5 - (-5)), 0))

from skimage.filters import gaussian
def segmental_correction(image, masks, flux_normalisation=gaussian_normalisation, blur=gaussian):
    from General_HelperFunctions import decompose_masks
    from numpy import stack, zeros_like, uint8

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    R1, G1 ,B1 = zeros_like(R), zeros_like(G), zeros_like(B)

    if flux_normalisation is not None:
        transformation = lambda x: ZtoRGB(flux_normalisation(x))
    else:
        transformation = lambda x: x

    for mask in decompose_masks(masks, reduce=False):
        R1[mask] = blur(transformation(R[mask]), preserve_range=True)
        G1[mask] = blur(transformation(G[mask]), preserve_range=True)
        B1[mask] = blur(transformation(B[mask]), preserve_range=True)

    return stack([R1, G1, B1], axis=2).astype(uint8)

def erodeNtimes(mask, N=1):
    from skimage.morphology import binary_erosion
    if N == 1: return binary_erosion(mask)
    return erodeNtimes(binary_erosion(mask), N=N-1)

def detect_outline(image):
    
    image = image.astype(bool)
    struct = np.ones((3, 3), dtype=bool)
    eroded_image = binary_erosion(image, structure=struct)
    outline = image ^ eroded_image
    
    return outline

### Classes

class Segment:
    def __init__(self, cell_rp, nucleus_rp, segment_order:int=0, bq:float=0.25):
        from numpy import where, nan, clip, exp, pi, quantile, stack, zeros_like, isnan, uint8
        from mahotas.features import haralick
        from pathtest import main_pathAnalysis

        self.cell_rp = cell_rp
        self.nucleus_rp = nucleus_rp
        self.segment_order = segment_order

        # Centre and neighbour

        self.x0, self.y0, self.x1, self.y1 = self.nucleus_rp.bbox
        self.xc, self.yc = self.nucleus_rp.centroid_local[0] + self.x0, self.nucleus_rp.centroid_local[1] + self.y0
        self.centre = (self.xc, self.yc)
        self.nearest_neighbour = 0

        # Colour channels
        self.cellR = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,0], nan)
        self.cellB = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,2], nan)
        self.nucleusR = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,0], nan)
        self.nucleusB = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,2], nan)

        cellRGB = stack([self.cellR, self.cellB, self.cellB], axis=2)
        nucleusRGB = stack([self.nucleusR, self.nucleusB, self.nucleusB], axis=2)

        cellRGB[isnan(cellRGB)] = 0
        nucleusRGB[isnan(nucleusRGB)] = 0

        self.cellRGB = uint8(cellRGB)
        self.nucleusRGB = uint8(nucleusRGB)

        # Binary channels
        self.binary_q = lambda q: clip((self.nucleusR > quantile(self.nucleusR[self.nucleus_rp.image], q)).astype(int) + detect_outline(self.nucleus_rp.image), 0, 1)
        self.binary_q50 = self.binary_q(bq)

        # Haralick features

        self.haralickR = haralick(self.nucleus_rp.intensity_image[:,:,0], ignore_zeros=True).mean(0)
        self.haralickB = haralick(self.nucleus_rp.intensity_image[:,:,2], ignore_zeros=True).mean(0)

        # Possible paths

        #self.paths = main_pathAnalysis(self.binary_q50)

        # Functions

        self.quantile = quantile
        self.gaussian = lambda x, mu=128, sigma=25.4: exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * pi * sigma**2)**0.5


    def show_segments(self, vmin=0, vmax=255):
        from matplotlib.pyplot import subplots, tight_layout, show, colorbar

        fig, axes = subplots(2, 2, figsize=(12,8), dpi=300)
        
        im = axes[0,0].imshow(self.cellR, origin='lower', extent=self.cell_rp.bbox, 
                              cmap='Reds', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[0,0], label='Cell redness', shrink=0.7)
        im = axes[0,1].imshow(self.cellB, origin='lower', extent=self.cell_rp.bbox, 
                              cmap='Blues', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[0,1], label='Cell blueness', shrink=0.7)
        im = axes[1,0].imshow(self.nucleusR, origin='lower', extent=self.nucleus_rp.bbox, 
                              cmap='Reds', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[1,0], label='Nucleus redness', shrink=0.7)
        im = axes[1,1].imshow(self.nucleusB, origin='lower', extent=self.nucleus_rp.bbox, 
                              cmap='Blues', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[1,1], label='Nucleus blueness', shrink=0.7)

        tight_layout()
        show()

    ### Regionprops
        
    def getNucleusArea(self): return self.nucleus_rp.area
    def getCellArea(self): return self.cell_rp.area

    def getNucleusAreaBbox(self): return self.nucleus_rp.area_bbox
    def getCellAreaBbox(self): return self.cell_rp.area_bbox

    def getNucleusAreaConvex(self): return self.nucleus_rp.area_convex
    def getCellAreaConvex(self): return self.cell_rp.area_convex

    def getNucleusAxisMajor(self): return self.nucleus_rp.axis_major_length
    def getCellAxisMajor(self): return self.cell_rp.axis_major_length

    def getNucleusAxisMinor(self): return self.nucleus_rp.axis_minor_length
    def getCellAxisMinor(self): return self.cell_rp.axis_minor_length

    def getNucleusEcc(self): return self.nucleus_rp.eccentricity
    def getCellEcc(self): return self.cell_rp.eccentricity

    def getNucleusDiam(self): return self.nucleus_rp.equivalent_diameter_area
    def getCellDiam(self): return self.cell_rp.equivalent_diameter_area

    def getNucleusFeretDiam(self): return self.nucleus_rp.feret_diameter_max
    def getCellFeretDiam(self): return self.cell_rp.feret_diameter_max

    def getNucleusMaxR(self): return self.nucleusR[self.nucleus_rp.image].max()
    def getCellMaxR(self): return self.cellR[self.cell_rp.image].max()

    def getNucleusMinR(self): return self.nucleusR[self.nucleus_rp.image].min()
    def getCellMinR(self): return self.cellR[self.cell_rp.image].min()

    def getNucleusStdR(self): return self.nucleusR[self.nucleus_rp.image].std()
    def getCellStdR(self): return self.cellR[self.cell_rp.image].std()

    def getNucleusSolidity(self): return self.nucleus_rp.solidity
    def getCellSolidity(self): return self.cell_rp.solidity

    def getNucleusOri(self): return self.nucleus_rp.orientation
    def getCellOri(self): return self.cell_rp.orientation

    def getNucleusPerim(self): return self.nucleus_rp.perimeter
    def getCellPerim(self): return self.cell_rp.perimeter

    ### Haralick

    def getH1R(self): return self.haralickR[0]
    def getH2R(self): return self.haralickR[1]
    def getH3R(self): return self.haralickR[2]
    def getH4R(self): return self.haralickR[3]
    def getH5R(self): return self.haralickR[4]
    def getH6R(self): return self.haralickR[5]
    def getH7R(self): return self.haralickR[6]
    def getH8R(self): return self.haralickR[7]
    def getH9R(self): return self.haralickR[8]
    def getH10R(self): return self.haralickR[9]
    def getH11R(self): return self.haralickR[10]
    def getH12R(self): return self.haralickR[11]
    def getH13R(self): return self.haralickR[12]

    def getH1B(self): return self.haralickB[0]
    def getH2B(self): return self.haralickB[1]
    def getH3B(self): return self.haralickB[2]
    def getH4B(self): return self.haralickB[3]
    def getH5B(self): return self.haralickB[4]
    def getH6B(self): return self.haralickB[5]
    def getH7B(self): return self.haralickB[6]
    def getH8B(self): return self.haralickB[7]
    def getH9B(self): return self.haralickB[8]
    def getH10B(self): return self.haralickB[9]
    def getH11B(self): return self.haralickB[10]
    def getH12B(self): return self.haralickB[11]
    def getH13B(self): return self.haralickB[12]

    ### Custom 
    from math import pi

    def getNucleusRoundness(self, normalisation=4*pi): return normalisation * self.nucleus_rp.area / self.nucleus_rp.perimeter**2
    def getCellRoundness(self, normalisation=4*pi): return normalisation * self.cell_rp.area / self.cell_rp.perimeter**2

    def getNucleusFraction(self): return self.nucleus_rp.area / self.cell_rp.area

    def getSegmentOrder(self): return self.segment_order
    
    def getEnergyR(self): return (self.nucleusR[self.nucleus_rp.image]**2).sum()
    def getEnergyB(self): return (self.nucleusB[self.nucleus_rp.image]**2).sum()

    def getEdgeFluxR(self, N=5): return self.nucleusR[erodeNtimes(self.nucleus_rp.image, N=N)].mean()

    def getNObjects(self):
        from skimage.measure import label
        return label(self.binary_q50, return_num=True, connectivity=2)[1]
    
    def getEuler(self):
        from skimage.measure import euler_number
        return euler_number(self.binary_q50, connectivity=2)
    
    def getNHoles(self): return self.getNObjects() - self.getEuler()

    def getAreaOfHoles(self): return (self.binary_q50^self.nucleus_rp.image).astype(int).sum() / self.nucleus_rp.area

    def getEntropyR(self):
        from skimage.measure import shannon_entropy
        return shannon_entropy(self.nucleusR)
    
    def getEntropyB(self):
        from skimage.measure import shannon_entropy
        return shannon_entropy(self.nucleusB)
    
    def getLLH(self):
        from numpy import log, mean
        return -mean(log(self.gaussian(self.nucleusR[self.nucleus_rp.image])))
    
    def getNearestN(self): return self.nearest_neighbour / self.nucleus_rp.equivalent_diameter_area

    def getQ05(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.05)

    def getQ25(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.25)

    def getQ50(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.50)

    def getQ75(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.75)

    def getQ95(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.95)

    def getSymR(self): return abs(self.quantile(self.nucleusR[self.nucleus_rp.image], 0.5) - self.nucleusR[self.nucleus_rp.image].mean())

    def getSymB(self): return abs(self.quantile(self.nucleusB[self.nucleus_rp.image], 0.5) - self.nucleusB[self.nucleus_rp.image].mean())

    #def getTotalPaths(self): return self.paths[0]
    #def getPathLength(self): return self.paths[1]
    
    ### Retrieving methods

    def retrieveFeatureNames(self): return [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("get")]

    def retrieveFeatures(self): return [getattr(self, func)() for func in self.retrieveFeatureNames()]

class Masks:
    def __init__(self, fname, type='control', flux_normalisation=gaussian_normalisation, bq:float=0.25):
        from cellpose.io import imread
        from numpy import load, arange, meshgrid
        from skimage.measure import regionprops
        from General_HelperFunctions import get_mask_levels, get_generations

        self.type = type
        if self.type == 'control':
            image_dir = '../control_images/'
            mask_dir = 'segments/control/'
        else:
            image_dir = '../drug_images/'
            mask_dir = 'segments/drug/'

        # Load image and its masks
        self.image = imread(image_dir + fname)
        data = load(mask_dir + fname + '.npy')
        self.cells, self.nuclei = data[:,:,0], data[:,:,1]
        self.levels = get_mask_levels(self.nuclei)
        self.generations = get_generations(self.levels)

        # Correct images
        self.cell_images = segmental_correction(self.image, self.cells, flux_normalisation=flux_normalisation)
        self.nucleus_images = segmental_correction(self.image, self.nuclei, flux_normalisation=flux_normalisation)

        # Genprops
        self.cell_regionprops = regionprops(self.cells, self.cell_images)
        self.nucleus_regionprops = regionprops(self.nuclei, self.nucleus_images)

        self.segments = []
        points = []
        for cell_rp, nucleus_rp, generation in zip(self.cell_regionprops, self.nucleus_regionprops, self.generations):
            seg = Segment(cell_rp, nucleus_rp, segment_order=generation, bq=bq)
            points.append(seg.centre)
            self.segments.append(seg)

        distances = nearest_neighbour(points)
        for seg, d in zip(self.segments, distances):
            seg.nearest_neighbour = d

    def getDataFrame(self):
        from pandas import DataFrame
        df = DataFrame()

        data = {}
        for label in self.segments[0].retrieveFeatureNames():
            data[label] = []

        if self.type == 'control':
            data['label'] = [0] * len(self.segments)
        else:
            data['label'] = [1] * len(self.segments)

        images_R = []; images_B = []
        for mask in self.segments:
            images_R.append(mask.nucleusR)
            images_B.append(mask.nucleusB)
            for label, val in zip(mask.retrieveFeatureNames(), mask.retrieveFeatures()):
                data[label].append(val)

        df = df.from_dict(data)
        self.df = df
        self.images_R = images_R
        self.images_B = images_B
        return self.df
    
###
    
def inverse_quantile(arr):
    from scipy.stats import percentileofscore
    return percentileofscore(arr, arr) / 100

from sklearn.preprocessing import StandardScaler
class Dataset:
    def __init__(self, control_paths, drug_paths, scaler=StandardScaler, flux_normalisation=gaussian_normalisation, bq:float=0.25):
        from pandas import concat
        from tqdm import tqdm

        self.control_paths = control_paths
        self.drug_paths = drug_paths

        # Create list of Masks objects
        print("Instantiating masks...")
        self.masks = []
        loop = tqdm(zip(self.control_paths + self.drug_paths, len(self.control_paths) * ['control'] + len(self.drug_paths) * ['drug']))
        for fname, type in loop:
            self.masks.append(Masks(fname.split('/')[-1], type=type, flux_normalisation=flux_normalisation, bq=bq))

        # Combined feature dataframes from each object
        print("Retrieving features...")
        dfs = []
        self.images_R = []
        self.images_B = []
        for mask in tqdm(self.masks):
            df = mask.getDataFrame()
            dfs.append(df)
            self.images_R += mask.images_R
            self.images_B += mask.images_B

        self.df = concat(dfs)
        self.feature_names = self.masks[0].segments[0].retrieveFeatureNames()
        self.feature_df = self.df[self.feature_names]

        # X-matrices
        self.X = self.feature_df.to_numpy()
        self.scaler = scaler().fit(self.X)
        self.X_scaled = self.scaler.transform(self.X)
        self.y = self.df['label'].to_numpy()

        self.X_reduced = None
        self.X_control_reduced = None
        self.X_drug_reduced = None

        self.is_control = (self.y==0)

        self.control_images_R = [self.images_R[i] for i, b in enumerate(self.is_control) if b]
        self.control_images_B = [self.images_B[i] for (i, b) in enumerate(self.is_control) if b]
        self.drug_images_R = [self.images_R[i] for i, b in enumerate(self.is_control) if not b]
        self.drug_images_B = [self.images_B[i] for (i, b) in enumerate(self.is_control) if not b]

    from sklearn.decomposition import PCA
    def performDimReduction(self, n_components:int=2, Algo=PCA):
        self.n_components = n_components
        algo = Algo(n_components=self.n_components)

        self.X_control = self.X_scaled[self.is_control,:]
        self.X_drug = self.X_scaled[~self.is_control,:]

        self.X_control_reduced = algo.fit_transform(self.X_control)
        self.X_drug_reduced = algo.fit_transform(self.X_drug)
        self.X_reduced = algo.fit_transform(self.X_scaled)

    def makeKDE(self, show_plot:bool=False, save_to=None, resolution:int=100, threshold=0):
        from scipy.stats import gaussian_kde
        from numpy import append, meshgrid, linspace, stack, quantile
        from matplotlib.pyplot import subplots, tight_layout, show, savefig, colorbar
        from matplotlib.colors import ListedColormap, BoundaryNorm

        if self.X_control_reduced is None:
            print('Run dimensionality reduction first!..')
            return
        
        if self.X_control_reduced.shape[0] == self.n_components: flip = lambda x: x
        else: flip = lambda x: x.T

        kde_control = gaussian_kde(flip(self.X_control_reduced))
        kde_drug = gaussian_kde(flip(self.X_drug_reduced))
        kde_control_combined = gaussian_kde(flip(self.X_reduced[self.is_control,:]))
        kde_drug_combined = gaussian_kde(flip(self.X_reduced[~self.is_control,:]))

        control_LLHs = kde_control.logpdf(flip(self.X_control_reduced))
        drug_LLHs = kde_drug.logpdf(flip(self.X_drug_reduced))

        self.control_qLLHs = inverse_quantile(control_LLHs)
        self.drug_qLLHs = inverse_quantile(drug_LLHs)

        self.df['Similarity'] = append(self.control_qLLHs, self.drug_qLLHs)

        if show_plot:
            xs, ys = linspace(-15, 15, resolution, endpoint=True), linspace(-15, 15, resolution, endpoint=True)
            xg, yg = meshgrid(xs, ys, indexing='ij')

            zg1 = kde_control.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)
            zg2 = kde_drug.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)
            zg3 = kde_control_combined.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution) - kde_drug_combined.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)

            qs = [0, 0.05, 0.1, 0.25, 0.5, 1]

            b1 = quantile(control_LLHs, qs); b2 = quantile(drug_LLHs, qs)
            l1 = b1[1:-1]; l2 = b2[1:-1]
            b3 = [-100, -10, -5, -3, -1, 0, 1, 3, 5, 10, 100]
            l3 = b3[1:-1]
            cmap1 = ListedColormap(['navy', 'blue', 'dodgerblue', 'deepskyblue', 'skyblue'])
            norm1 = BoundaryNorm(b1, cmap1.N)
            cmap2 = ListedColormap(['firebrick', 'crimson', 'red', 'orangered', 'darkorange'])
            norm2 = BoundaryNorm(b2, cmap2.N)
            cmap3 = ListedColormap(['firebrick', 'crimson', 'red', 'orangered', 'darkorange', 'skyblue', 'deepskyblue', 'dodgerblue', 'blue', 'navy'])
            norm3 = BoundaryNorm(b3, cmap3.N)

            fig, (l, r, b) = subplots(1, 3, figsize=(16, 4), dpi=300)

            l.set_title('Control'); r.set_title('Drug'); b.set_title('Combined')

            im1 = l.imshow(zg1.T, origin='lower', cmap=cmap1, norm=norm1, extent=(-15, 15, -15, 15))
            im2 = r.imshow(zg2.T, origin='lower', cmap=cmap2, norm=norm2, extent=(-15, 15, -15, 15))
            im3 = b.imshow(zg3.T, origin='lower', cmap=cmap3, norm=norm3, extent=(-15, 15, -15, 15))

            cs1 = l.contour(xg.T, yg.T, zg1.T, levels=l1, colors='k')
            cs2 = r.contour(xg.T, yg.T, zg2.T, levels=l2, colors='k')
            cs3 = b.contour(xg.T, yg.T, zg3.T, levels=l3, colors='k')

            cbar1 = colorbar(im1, ax=l, label='Quantile')
            cbar2 = colorbar(im2, ax=r, label='Quantile')
            cbar3 = colorbar(im3, ax=b, label='Log Prob. Ratio')

            cbar1.ax.set_yticklabels(qs)
            cbar2.ax.set_yticklabels(qs)
            cbar3.ax.set_yticklabels(b3)

            control_below = (self.control_qLLHs < threshold)
            drug_below = (self.drug_qLLHs < threshold)

            l.scatter(self.X_control_reduced[control_below,0], self.X_control_reduced[control_below,1], zorder=2, c='fuchsia', s=0.5)
            l.scatter(self.X_control_reduced[~control_below,0], self.X_control_reduced[~control_below,1], zorder=1, c='white', s=0.5)

            r.scatter(self.X_drug_reduced[drug_below,0], self.X_drug_reduced[drug_below,1], zorder=2, c='gold', s=0.5)
            r.scatter(self.X_drug_reduced[~drug_below,0], self.X_drug_reduced[~drug_below,1], zorder=1, c='black', s=0.5)

            b.scatter(self.X_reduced[self.is_control,:][control_below,0], self.X_reduced[self.is_control,:][control_below,1], zorder=2, c='fuchsia', s=0.5)
            b.scatter(self.X_reduced[self.is_control,:][~control_below,0], self.X_reduced[self.is_control,:][~control_below,1], zorder=1, c='white', s=0.5)
            b.scatter(self.X_reduced[~self.is_control,:][drug_below,0], self.X_reduced[~self.is_control,:][drug_below,1], zorder=2, c='gold', s=0.5)
            b.scatter(self.X_reduced[~self.is_control,:][~drug_below,0], self.X_reduced[~self.is_control,:][~drug_below,1], zorder=1, c='black', s=0.5)

            for ax in [l, r, b]:
                ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
                ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')

            tight_layout()
            if save_to is not None:
                savefig(save_to, dpi=300, bbox_inches='tight')
            show()

    def makeSelectionKDE(self, q_control=0.05, q_drug=0.05):
        condition = (self.df['label'] == 0) * (self.df['Similarity'] >= q_control) + (self.df['label'] == 1) * (self.df['Similarity'] >= q_drug)
        df_reduced = self.df[condition]
        return df_reduced[self.feature_names + ['label']]

   