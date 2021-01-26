# Pre-processing CXR data for federated learning DeepMed Platform

## Requirements: Python packages
`
pydicom
sklearn
imageio
numpy
scipy
pillow
pandas
matplotlib
`


We recommend using a [pip virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

    #(If needed) install pip and virtualenv:
    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user virtualenv

    # create virtual environment
    python3 -m venv deepmed_env

    # activate virtual environment
    source deepmed_env/bin/activate


## 1. CXR preprocessing
Save dicom files of CXRs with the following structure:

```bash
DICOM_ROOT
├── CLASS_NAME_1
│   ├── ID_1.dcm
│   └── ID_2.dcm
├── CLASS_NAME_2
│   ├── ID_3.dcm
│   └── ID_4.dcm
└── CLASS_NAME_3
    ├── ID_5.dcm
    └── ID_6.dcm
```

Convert dicom files to png image files:

`
python3 dcm2png.py --input_dir DICOM_ROOT --output_dir OUTPUT_FOLDER
`

png images will be saved as `OUTPUT_FOLDER/png/CLASS_NAME/*.png`

## 2. Generate dataset json file

This script requires a json file "CLASS_MAP.json" which contains mapping from CLASS_NAME into integer. The example content of said file is shown here.

```{'normal': 0, 'covid': 1, 'pneumonia': 2}```

Split dataset with stratified sampling and create dataset json file that will be used in federated learning by CLARA Train SDK:

`
python3 json_gen.py --png_dir OUTPUT_FOLDER/png --map_json CLASS_MAP.json --output_dir OUTPUT_FOLDER/
`

The dataset json file will be saved as OUTPUT_FOLDER/datalist.json

## 3. Compute image intensitiy distribution

Compute the image intensity distribution by using

    DATA_ROOT=/OUTPUT_FOLDER/png
    DATALIST_PATH=/OUTPUT_FOLDER/datalist.json
    OUTPUT_DIR=./distributions
    ./compute_stats.sh ${DATA_ROOT} ${DATALIST_PATH} ${OUTPUT_DIR}
