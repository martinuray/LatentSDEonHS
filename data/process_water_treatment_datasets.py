import argparse
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1))
    normalizer.fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret

# downsample by 10
def downsample(data, labels=None, down_len=10):
    np_data = np.array(data)
    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()

    # see [Deng and Hooi, 2021]
    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)
    d_data = d_data.transpose()

    if labels is None:
        d_labels = None
    else:
        d_labels = labels.values[:down_time_len*down_len].reshape(-1, down_len)
        # if exist anomalies, then this sample is abnormal
        # implemented unlike [Deng and Hooi, 2021]; they say take the mode
        d_labels = np.round(np.max(d_labels, axis=1))
        d_labels = d_labels.tolist()

    return d_data.tolist(), d_labels

def get_parser():
    parser = argparse.ArgumentParser(description="Water Treatment Data Preprocessor")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["SWaT", "WaDiv1", "WaDiv2"],
        help="Dataset to process"
    )
    args = parser.parse_args()
    return args


def main(args):



    if "WaDi" in args.dataset:
        subpath = "v1"
        if "v2" in args.dataset:
            subpath = "v2"
        train_df = pd.read_csv(f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/rawraw/{subpath}/WADI_14days.csv')
        test_df = pd.read_csv(f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/rawraw/{subpath}/WADI_attackdata_labelled.csv',
                              skiprows=1)
        test_labels = test_df['Attack LABLE (1:No Attack, -1:Attack)'].map(
            {1: 0, -1: 1})
        test_df = test_df.drop(
            columns=['Attack LABLE (1:No Attack, -1:Attack)'])
        train_df = train_df.iloc[:, 3:]
        test_df = test_df.iloc[:, 3:]

    else:
        train_df = pd.read_csv(f'data_dir/{args.dataset}/raw/train_raw.csv')
        test_df = pd.read_csv(f'data_dir/{args.dataset}/raw/test_raw.csv')
        test_labels = pd.read_csv(f'data_dir/{args.dataset}/raw/labels_raw.csv')
        test_labels = test_labels['labels']

    train_df = train_df.fillna(train_df.mean())
    test_df = test_df.fillna(test_df.mean())
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # trim column names
    train_df = train_df.rename(columns=lambda x: x.strip())
    test_df = test_df.rename(columns=lambda x: x.strip())
    if args.dataset == "WaDi":
        cols = [x[46:] for x in train_df.columns]  # remove column name prefixes
        train_df.columns = cols
        test_df.columns = cols

    x_train, x_test = norm(train_df.values, test_df.values)

    d_train_x, _ = downsample(x_train)
    d_test_x, d_test_labels = downsample(x_test, test_labels)

    train_df = pd.DataFrame(d_train_x, columns=train_df.columns)
    test_df = pd.DataFrame(d_test_x, columns=test_df.columns)
    test_labels_df = pd.DataFrame(d_test_labels, columns=['labels'])

    # implemented as [Deng and Hooi, 2021]; they say to ignore the first 2160
    # samples, as during the first 5-6 hours the system needs to reach a
    # stabilizing state; 2160 equals the subsampled duration of 6 hours
    train_df = train_df.iloc[2160:]

    def store_file(df_, path_, file_):
        df_.to_csv(os.path.join(path_, f'{file_}.csv'), index=False)
        np.save(os.path.join(path_, f'{file_}.npy'), df_.to_numpy())

    store_file(train_df, f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/', 'train')
    store_file(test_df, f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/', 'test')
    store_file(test_labels_df, f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/', 'labels')

    with open(f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/list.txt', 'w') as f:
        for col in train_df.columns:
            f.write(col+'\n')


if __name__ == '__main__':
    args = get_parser()
    main(args)
