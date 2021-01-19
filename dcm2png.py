import pydicom
import numpy as np
import glob
import os
import scipy.ndimage
import imageio
from multiprocessing import Pool
from functools import partial
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description = 'CXR Preprocess')
parser.add_argument('--input_dir', type=str, required=True, help="input CXR dicom directory")
parser.add_argument('--output_dir', type=str, required=True, help="output directory")


def read_general(dcm):
    img = dcm.pixel_array
    
    if hasattr(dcm, 'VOILUTSequence'):
        img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
    
    if hasattr(dcm, 'PixelIntensityRelationshipSign'):
        if dcm.PixelIntensityRelationshipSign == 1 and dcm.PixelIntensityRelationship == 'LIN':
            img = -img
    elif hasattr(dcm, 'PresentationLUTShape'):
        if dcm.PresentationLUTShape == 'INVERSE':
            img = -img
    
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        vmin = dcm.WindowCenter - dcm.WindowWidth / 2
        vmax = dcm.WindowCenter + dcm.WindowWidth / 2
    else:
        vmin = img.min()
        vmax = img.max()
    
    return (img - vmin) / (vmax - vmin)

def read_fuji_iray(dcm):
    img = dcm.pixel_array
    
    if hasattr(dcm, 'VOILUTSequence'):
        img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
    
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        vmin = dcm.WindowCenter - dcm.WindowWidth / 2
        vmax = dcm.WindowCenter + dcm.WindowWidth / 2
    else:
        vmin = img.min()
        vmax = img.max()
    
    return 1 - (img - vmin) / (vmax - vmin)

def read_swissray(dcm):
    img = dcm.pixel_array
    
    if hasattr(dcm, 'VOILUTSequence'):
        img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
    
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        vmin = dcm.WindowCenter - dcm.WindowWidth / 2
        vmax = dcm.WindowCenter + dcm.WindowWidth / 2
    else:
        vmin = img.min()
        vmax = img.max()
    
    return (img - vmin) / (vmax - vmin)

def read_ge(dcm):
    '''
    Use window center, window width if possible, otherwise directly apply LUT
    '''
    img = dcm.pixel_array
    
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):        
        # use the soft window to be consistent with other vendors    
        if dcm.WindowWidth[2] >= (img.max() - img.min()):
            vmin = img.min()
            vmax = img.max()
        else:
            vmin = dcm.WindowCenter[2] - dcm.WindowWidth[2] / 2
            vmax = dcm.WindowCenter[2] + dcm.WindowWidth[2] / 2
        img = (img - vmin) / (vmax - vmin)
    else:
        img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, dcm)
        img = (img - img.min()) / (img.max() - img.min())
    
    return img

def read_by_manufacturer(dcm):
    if not hasattr(dcm, 'PixelData'):
        return None
    
    manufacturer = dcm.Manufacturer.lower()
    if 'altamont' in manufacturer or 'lexmark' in manufacturer or 'pacsgear' in manufacturer:
        # these are texts
        return None
    elif 'ge health' in manufacturer or 'ge medical' in manufacturer:
        return read_ge(dcm)
    elif 'fuji' in manufacturer or 'iray' in manufacturer:
        return read_fuji_iray(dcm)
    elif 'swissray' in manufacturer:
        return read_swissray(dcm)
    else:
        return read_general(dcm)

def read_single_file(filename):
    dcm = pydicom.dcmread(filename)
    if not hasattr(dcm, 'ImageType'):
        return None
    img = read_by_manufacturer(dcm)
    return img

def process_folder(filenames, class_name, png_dir, img_shape):
    for filename in tqdm(filenames):
        img = read_single_file(filename)
        if img is None: continue
        
        # save image
        # apply a prefilter to the image when downsampling
        zoom = np.array(img_shape) / np.array(img.shape)
        stds = 1 / zoom / 4
        img = scipy.ndimage.filters.gaussian_filter1d(img, stds[0], axis=0)
        img = scipy.ndimage.filters.gaussian_filter1d(img, stds[1], axis=1)
        img = scipy.ndimage.interpolation.zoom(img, zoom)
        # convert to rgb
        img[img < 0] = 0
        img[img > 1] = 1
        img = np.tile(img[..., np.newaxis], (1,1,3)) * 255
        output_filename = filename.split('/')[-1].replace('.dcm','.png')
        imageio.imwrite(os.path.join(png_dir, class_name, output_filename), img.astype(np.uint8))
    
def main():
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    png_dir = os.path.join(output_dir,"png")
    img_shape = (512, 512)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    filepath_list = glob.glob(input_dir+'/*/*/*.dcm')
    folder_dct = dict()
    for filepath in filepath_list:
        pred_class = filepath.split('/')[-3]
        if pred_class not in folder_dct: folder_dct[pred_class] = list()
        folder_dct[pred_class].append(filepath)
    for class_name, filenames in folder_dct.items():
        if not os.path.exists(os.path.join(png_dir, class_name)):
            os.makedirs(os.path.join(png_dir, class_name))
        print("processing %s folder" % (class_name))
        process_folder(filenames, class_name, png_dir, img_shape)


if __name__ == '__main__':
    main()