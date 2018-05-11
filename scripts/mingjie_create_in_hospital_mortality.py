import os
import argparse
import pandas as pd
from datetime import datetime
import random


def process_basic_data(args, partition, episode_dirname='basic'):
    df = None
    df_path = os.path.join(args.output_path, partition + '_listfile.csv')
    input_dir = os.path.join(args.root_path, partition)
    patient_dirnames = list(filter(str.isdigit, os.listdir(input_dir)))

    try:
        output_dir = os.path.join(args.output_path, partition)
        episode_outdir = os.path.join(output_dir, episode_dirname)
        os.makedirs(episode_outdir)
    except FileExistsError:
        pass

    for patient_index, patient in enumerate(patient_dirnames):
        patient_dir = os.path.join(input_dir, patient)
        ts_fnames = list(filter(lambda x: x.find("timeseries") != -1,
                                os.listdir(patient_dir)))

        for ts_fname in ts_fnames:
            lb_fname = ts_fname.replace("_timeseries", "")

            label_df = pd.read_csv(os.path.join(patient_dir, lb_fname))
            ts_df = pd.read_csv(os.path.join(patient_dir, ts_fname))

            # Quality check ---------------------------
            if len(label_df) == 0 or len(ts_df) == 0:
                print('Empty label df or ts df', patient, ts_fname)
                continue

            assert len(label_df == 1)
            label_df = label_df.to_dict(orient='records')[0]
            los = label_df['Length of Stay'] * 24  # length of stay in hours

            if pd.isnull(los):
                print("length of stay is missing", patient, ts_fname)
                continue
            elif label_df["Mortality"] == pd.isnull(label_df["Deathtime"]):
                print('Unmatch mortality label and deathtime', patient_dir)
                continue

            # Copy over time series data --------------
            ts_df.to_csv(os.path.join(episode_outdir, patient + "_" + ts_fname))

            # Collect time invarient information ------
            rel_death_hours = None
            if not pd.isnull(label_df["Deathtime"]) and not pd.isnull(label_df["Intime"]):
                fmt = '%Y-%m-%d %H:%M:%S'
                tdelta = datetime.strptime(label_df["Deathtime"], fmt) -\
                         datetime.strptime(label_df["Intime"], fmt)
                rel_death_hours = tdelta.days * 24 + tdelta.seconds / 60 / 60

            basis_keys = ['Icustay', 'Intime', 'Outtime', 'Deathtime',
                    'Mortality', 'Ethnicity', 'Gender', 'Age', 'Height', 'Weight']
            diagnosis_keys = [k for k in label_df.keys() if k.startswith('Diagnosis ')]

            new_df = {k: label_df[k] for k in basis_keys + diagnosis_keys}
            new_df['Subject'] = patient
            new_df['Real Death Hours'] = rel_death_hours
            new_df['Length of Stay'] = los
            new_df['Episode'] = ts_fname
            new_df = pd.DataFrame(new_df, index=[0])
            df = df.append(new_df, ignore_index=True) if df is not None else new_df

        if ((patient_index + 1) % 100 == 0):
            print("\rprocessed {} / {} patients".format(patient_index + 1,
                                                        len(patient_dirnames)))

    print('Number of encounters: {}'.format(len(df)))
    if partition == "train":
        df = df.sample(frac=1)
    if partition == "test":
        df = df.sort_values('Icustay', ascending=False)

    if df is not None:
        df.to_csv(df_path)


def process_extra_data(args, partition, episode_dirname):
    df_path = os.path.join(args.output_path, partition + '_listfile.csv')
    df = pd.read_csv(df_path)
    df[episode_dirname] = 0
    patients = df['Subject'].tolist()
    input_dir = os.path.join(args.root_path, partition)

    try:
        output_dir = os.path.join(args.output_path, partition)
        episode_outdir = os.path.join(output_dir, episode_dirname)
        os.makedirs(episode_outdir)
    except FileExistsError:
        pass

    for patient in patients:
        patient_dir = os.path.join(input_dir, str(patient))
        episode_dir = os.path.join(patient_dir, episode_dirname)

        if not os.path.isdir(episode_dir):
            continue

        ts_fnames = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(episode_dir)))

        for ts_fname in ts_fnames:
            lb_fname = ts_fname.replace("_timeseries", "")

            # Copy over time series data
            ts_df = pd.read_csv(os.path.join(episode_dir, ts_fname))
            lb_df = pd.read_csv(os.path.join(episode_dir, lb_fname))

            df.loc[df['Icustay'] == lb_df['Icustay'].tolist()[0], episode_dirname] = 1
            ts_df.to_csv(os.path.join(episode_outdir, str(patient) + "_" + ts_fname))

    df.to_csv(df_path)


if __name__ == '__main__':
    random.seed(49297)

    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('-root_path', type=str,
                        default='/Users/MJ/Desktop/temp/root',
                        help="Path to root folder containing train and test sets.")
    parser.add_argument('-output_path', type=str,
                        default='/Users/MJ/Desktop/temp/output',
                        help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    process_basic_data(args, "train")
    process_extra_data(args, 'train', episode_dirname='drug_level')
    process_basic_data(args, "test")
    process_extra_data(args, 'test', episode_dirname='drug_level')