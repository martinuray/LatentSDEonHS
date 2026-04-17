import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import MinMaxScaler


def reshape_data(data, window_length, remove_zero_column=True):
    if type(data) != np.ndarray:
        data_np = data.to_numpy()
    else:
        data_np = data

    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)

    mod = data_np.shape[0] % window_length
    if type(data) == pd.DataFrame and 'labels' in data.columns:
        # only for labels
        if mod > 0:
            data_np = data_np.squeeze()[:-mod]
        shaped = data_np.reshape(-1, window_length)
    else:
        # to ignore index col
        if mod > 0:
            data_np = data_np[:-mod]
        if remove_zero_column:
            data_np = data_np[:, 1:] # to ignore index col

        shaped = data_np.reshape(-1, window_length, data_np.shape[1])
    return shaped

def norm(train_df, test_df):
    normalizer = MinMaxScaler(feature_range=(0, 1))
    normalizer.fit(train_df) # scale training data to [0,1] range

    train_df_normed, test_df_normed = train_df.copy(), test_df.copy()
    train_df_normed[train_df.columns] = normalizer.transform(train_df)
    test_df_normed[test_df.columns] = normalizer.transform(test_df)
    return train_df_normed, test_df_normed

# downsample by 10
def downsample(data_df, labels=None, down_len:int=10):
    """
    Args:
        data_df: dataframe with data
        labels: dataframe with labels
        down_len: the factor, by which the data is downsampled

    Returns:

    """
    return data_df, labels
    np_data = np.array(data_df)
    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    # see [Deng and Hooi, 2021]
    d_data = data_df #.groupby(data_df.index // 10).median()

    d_labels = None
    if labels is not None:
        d_labels = labels.groupby(labels.index // 10).max()
        # if exist anomalies, then this sample is abnormal
        # implemented unlike [Deng and Hooi, 2021]; they say take the mode


    return d_data, d_labels

def get_parser():
    parser = argparse.ArgumentParser(description="Water Treatment Data Preprocessor")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["SWaT", "WaDiv1", "WaDiv2"],
        help="Dataset to process"
    )

    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--window-overlap", type=float, default=0)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--make-validation-set", action=argparse.BooleanOptionalAction, default=False)
    arguments_ = parser.parse_args()
    return arguments_


def visualize_comparison(after_transform=None, before_transform=None,
                         num_viz:int=30, step_size:int=10000):

    for viz_idx in tqdm.tqdm(range(num_viz)):
        start_idx = viz_idx * step_size

        if start_idx > before_transform.shape[0]:
            break

        n_rows = before_transform.shape[1] // 2
        fig, axs = plt.subplots(nrows=n_rows, ncols=2, sharex=True, figsize=(24, 14))
        axs = axs.flatten()

        for col_idx, column_name in enumerate(before_transform.columns):
            if col_idx == len(axs):
                break

            if before_transform is not None:
                axs[col_idx].plot(before_transform[column_name].iloc[start_idx:start_idx + step_size],
                                  label="raw")
            if after_transform is not None:
                axs[col_idx].plot(after_transform[column_name].iloc[start_idx:start_idx + step_size],
                                  '--', label="processed")
            axs[col_idx].set_title(column_name)
            axs[col_idx].legend()

        plt.ioff()
        plt.show()
        plt.close('all')


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
        train_df = pd.read_csv(f'data_dir/{args.dataset}/rawraw/train.csv')
        test_df = pd.read_csv(f'data_dir/{args.dataset}/rawraw/test.csv')
        test_labels = pd.read_csv(f'data_dir/{args.dataset}/rawraw/labels.csv')
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

    # remove columns, where there is only one value in either train or test set
    def _get_set_of_unique_values(df_):
        return set(df_.columns[df_.nunique() == 1].tolist())
    train_col_set = _get_set_of_unique_values(train_df)
    test_col_set = _get_set_of_unique_values(test_df)
    common_set = list(train_col_set.union(test_col_set))

    train_df = train_df.drop(common_set, axis=1)
    test_df = test_df.drop(common_set, axis=1)

    col_names_train = train_df.columns[train_df.nunique() <= 10]

    train_df_filtered, test_df_filtered = train_df, test_df

    # what normalize?
    train_df_normalized, test_df_normalized = norm(train_df_filtered, test_df_filtered)

    #train_df_filtered, _ = downsample(train_df_normalized, down_len=args.downsample_factor)
    train_df_filtered = train_df_normalized

    #test_df_filtered, test_labels = downsample(test_df_normalized, test_labels,
    #                                                down_len=args.downsample_factor)
    test_df_filtered, test_labels = test_df_normalized, test_labels

    if "WaDi" in args.dataset:
        df_test_labels = pd.DataFrame(test_labels)
        df_test_labels.columns = ['labels']
    else:
        df_test_labels = pd.DataFrame(test_labels, columns=['labels'])
#train_df_filtered = train_df_filtered.reset_index()

    if args_.debug:
        df_comparison_num_critical_features = pd.DataFrame({
            'before': train_df[col_names_train].nunique(),
            'after': train_df_filtered[col_names_train].nunique(),
        })

        print(df_comparison_num_critical_features)

        visualize_comparison(
            train_df_filtered[col_names_train],
            train_df[col_names_train].iloc[::10].reset_index().drop(['index'], axis=1),
            step_size = 200
        )

    # implemented as [Deng and Hooi, 2021]; they say to ignore the first 2160
    # samples, as during the first 5-6 hours the system needs to reach a
    # stabilizing state; 2160 equals the subsampled duration of 6 hours
    train_df_filtered = train_df_filtered.iloc[2160:]

    # TODO
    #if "WaDi" in self.data_kind:
    #    logging.info(f"Limiting WaDi Dataset to {60_000} samples, as per reference")
    #    # done as with QuoVadis, see
    #    # https://github.com/ssarfraz/QuoVadisTAD/blob/8e2de5a1574d1f8b2b669e2aa81a34fd92bd5b58/quovadis_tad/model_utils/model_def.py#L55
    #    raw_data_df = raw_data.iloc[:60_000]

    # process further
    train_np = reshape_data(train_df_filtered, args.window_length)
    test_np = reshape_data(test_df_filtered, args.window_length)
    labels_np = reshape_data(df_test_labels, args.window_length)


    def store_file(data, path_, file_):
        os.makedirs(path_, exist_ok=True)
        #df_.to_csv(os.path.join(path_, f'{file_}.csv'), index=False)
        np.save(os.path.join(path_, f'{file_}.npy'), data)

    if not args.debug:
        path_ = f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/'
        if args.make_validation_set:
            df_len = train_np.shape[0]
            indices = np.random.permutation(df_len)
            val_indices = indices[int(df_len * args.validation_split):]
            train_indices = indices[:int(df_len * args.validation_split)]

            store_file(train_np[train_indices], path_, 'train')
            store_file(train_np[val_indices], path_, 'val')
        else:
            store_file(train_np, path_, 'train')
        store_file(test_np, path_, 'test')
        store_file(labels_np, path_, 'labels')

        with open(f'data_dir/{args.dataset.replace("v1", "").replace("v2", "")}/raw/list.txt', 'w') as f:
            for col in train_df.columns:
                f.write(col+'\n')


if __name__ == '__main__':
    args_ = get_parser()
    print(args_)
    main(args_)
