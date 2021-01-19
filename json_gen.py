import glob
import os
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description = 'CXR Preprocess')
parser.add_argument('--png_dir', type=str, required=True, help="input CXR dicom directory")
parser.add_argument('--map_json', type=str, required=True, help="json file map class name to integer")
parser.add_argument('--output_dir', type=str, required=True, help="output directory for json file")

def dataframe_to_jsonformat(df, label_length):
    datalist = []
    for index, row in df.iterrows():
        item = dict()
        item['image'] = row['image']
        item['label'] = [0]*label_length
        item['label'][row['class']] = 1
        datalist.append(item)
    return datalist
    

def main():
    args = parser.parse_args()
    png_dir = args.png_dir
    map_json = args.map_json
    output_dir = args.output_dir
    class_names = sorted(os.listdir(png_dir))
    map_json_file = open(map_json)
    class_map = json.load(map_json_file)

    dct = dict()
    for class_name in class_names:
        for filename in os.listdir(os.path.join(png_dir,class_name)):
            dct[os.path.join(class_name,filename)] = class_name
    all_df = pd.DataFrame(dct.items(), columns=["image", "class"])
    all_df['class'] = all_df['class'].map(class_map)
    train_df, val_test_df = train_test_split(all_df, stratify=all_df['class'], test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(val_test_df, stratify = val_test_df['class'], test_size=0.5, random_state=42)

    data = {}
    data["label_format"] = [1]*len(class_names)
    data["training"] = dataframe_to_jsonformat(train_df, len(class_names))
    data["validation"] = dataframe_to_jsonformat(val_df, len(class_names))
    data["testing"] = dataframe_to_jsonformat(test_df, len(class_names))

    with open(os.path.join(output_dir, 'datalist.json'), 'w') as output_file:
        json.dump(data, output_file)


if __name__ == '__main__':
    main()