import os
import argparse
import pandas as pd
from datetime import datetime
import random

random.seed(49297)

parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
parser.add_argument('-root_path', type=str,
                    default='/Users/MJ/Desktop/temp/root',
                    help="Path to root folder containing train and test sets.")
parser.add_argument('-output_path', type=str,
                    default='/Users/MJ/Desktop/temp/output',
                    help="Directory where the created data should be stored.")
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


def process_partition(partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    df = None

    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            output_ts_filename = patient + "_" + ts_filename
            lb_filename = ts_filename.replace("_timeseries", "")

            label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            ts_df = pd.read_csv(os.path.join(patient_folder, ts_filename))

            # Quality check
            if len(label_df) == 0 or len(ts_df) == 0:
                print('Empty label df or ts df', patient, ts_filename)
                continue

            assert len(label_df == 1)
            label_df = label_df.to_dict(orient='records')[0]
            los = label_df['Length of Stay'] * 24  # length of stay in hours

            if pd.isnull(los):
                print("length of stay is missing", patient, ts_filename)
                continue
            elif label_df["Mortality"] == pd.isnull(label_df["Deathtime"]):
                print('Unmatch mortality label and deathtime', patient_folder)
                continue

            # Copy over time series data
            ts_df.to_csv(os.path.join(output_dir, output_ts_filename))

            # Collect time invarient information
            rel_death_hours = None
            if not pd.isnull(label_df["Deathtime"]) and not pd.isnull(label_df["Intime"]):
                fmt = '%Y-%m-%d %H:%M:%S'
                tdelta = datetime.strptime(label_df["Deathtime"], fmt) -\
                         datetime.strptime(label_df["Intime"], fmt)
                rel_death_hours = tdelta.days * 24 + tdelta.seconds / 60 / 60

            keys = ['Icustay', 'Intime', 'Outtime', 'Deathtime',
                    'Mortality', 'Ethnicity', 'Gender', 'Age', 'Height', 'Weight']
            new_df = {k: label_df[k] for k in keys}
            new_df['Filename'] = output_ts_filename
            new_df['Real Death Hours'] = rel_death_hours
            new_df['Length of Stay'] = los
            new_df = pd.DataFrame(new_df, index=[0])
            df = df.append(new_df, ignore_index=True) if df is not None else new_df

        if ((patient_index + 1) % 100 == 0):
            print("\rprocessed {} / {} patients".format(patient_index + 1, len(patients)))

    print('Number of encounters: {}'.format(len(df)))
    if partition == "train":
        df = df.sample(frac=1)
    if partition == "test":
        df = df.sort_values('Icustay', ascending=False)

    df.to_csv(os.path.join(output_dir, "listfile.csv"))

process_partition("train")
process_partition("test")
