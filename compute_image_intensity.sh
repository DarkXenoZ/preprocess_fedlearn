#!/usr/bin/env bash
DATA_ROOT=$1
DATALIST_PATH=$2
OUTPUT_DIR=$3

if test -z "${DATA_ROOT}"
then
      echo "Please the data root location of images and datalists"
      exit 1
fi
if test -z "${DATALIST_PATH}"
then
      echo "Please the filename of datalist"
      exit 1
fi
if test -z "${OUTPUT_DIR}"
then
      echo "Please the path to output directory"
      exit 1
fi

echo "PROCESSING ${DATALIST_PATH}"
python plot_histogram.py --data_root="${DATA_ROOT}" --dataset_json="${DATALIST_PATH}" --keys training,validation,testing --output_root="${OUTPUT_DIR}/histogram"
ret=$?
if [ $ret -ne 0 ]; then
   echo "THERE WAS AN ERROR WHEN PLOTTING HISTOGRAMS! PLEASE CHECK THE ABOVE OUTPUTS TO RESOLVE IT."
   exit 1
fi
