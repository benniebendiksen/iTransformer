import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Crypto(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='btcusdc_pca_components_44_proper_split.csv',
                 target='close', scale=True, timeenc=0, freq='15min'):
        # size [seq_len, label_len, pred_len]
        # info
        print(f"size is: {size}")
        if size == None:
            self.seq_len = 96  # 24 hours (96 * 15 min)
            self.label_len = 48
            self.pred_len = 4  # Predict 4 steps ahead
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.set_type = flag  # Instead of numeric mapping, use the string directly

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Handle timestamp vs date column naming
        if 'timestamp' in df_raw.columns and 'date' not in df_raw.columns:
            print(f"converting col name timestamp to date")
            df_raw = df_raw.rename(columns={'timestamp': 'date'})

            # For btcusdc_1d_historical.csv, parse human-readable dates
            try:
                df_raw['date'] = pd.to_datetime(df_raw['date'])
                print(f"Successfully parsed human-readable dates in 'date' column")
            except Exception as e:
                print(f"Warning: Failed to parse dates: {e}")

        # Convert dates to Unix timestamps if needed
        if 'date' in df_raw.columns:
            if not pd.api.types.is_numeric_dtype(df_raw['date']):
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df_raw['date']):
                        df_raw['date'] = pd.to_datetime(df_raw['date'])

                    df_raw['date_readable'] = df_raw['date'].dt.strftime('%Y-%m-%d')
                    df_raw['date'] = df_raw['date'].astype('int64') // 10 ** 9
                    print(f"Converted dates to Unix timestamps")
                except Exception as e:
                    print(f"Warning: Error converting dates to timestamps: {e}")

        # Verify target column exists
        if self.target not in df_raw.columns:
            print(
                f"Warning: Target column '{self.target}' not found. Available columns: {df_raw.columns.tolist()[:10]}")
            # For historical dataset, look for a close-like column
            possible_targets = ['close', 'Close', 'price', 'Price']
            for col in possible_targets:
                if col in df_raw.columns:
                    print(f"Using '{col}' as target instead of '{self.target}'")
                    self.target = col
                    break

        # Check if the dataset has a 'split' column
        has_split_column = 'split' in df_raw.columns

        # If using the new properly generated dataset with split column
        if has_split_column:
            # Use the split column to filter data
            train_data = df_raw[df_raw['split'] == 'train']
            val_data = df_raw[df_raw['split'] == 'val']
            test_data = df_raw[df_raw['split'] == 'test']

            # Remove the split column before feature processing
            df_raw = df_raw.drop(columns=['split'])
            train_data = train_data.drop(columns=['split'])
            val_data = val_data.drop(columns=['split'])
            test_data = test_data.drop(columns=['split'])

            # Set the appropriate dataset based on flag
            if self.set_type == 'train':
                active_data = train_data
            elif self.set_type == 'val':
                active_data = val_data
            else:  # 'test'
                active_data = test_data

            # Define borders
            border1 = 0
            border2 = len(active_data)

        else:
            # Original split method from before (percentage-based)
            # Split data into train, validation, test (70%, 10%, 20%)
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

            # Map set_type to numeric index for backward compatibility
            type_map = {'train': 0, 'val': 1, 'test': 2}
            set_type_idx = type_map[self.set_type]

            border1 = border1s[set_type_idx]
            border2 = border2s[set_type_idx]

            # For traditional split, active_data is the full dataset
            active_data = df_raw
            train_data = df_raw.iloc[:num_train]

        # Create binary labels for price change prediction
        # For each point, 1 if price 4 steps ahead > current price, 0 otherwise
        close_prices = df_raw[self.target].values
        binary_labels = np.zeros((len(close_prices), 1))

        for i in range(len(close_prices) - self.pred_len):
            future_price = close_prices[i + self.pred_len]
            current_price = close_prices[i]
            binary_labels[i, 0] = 1.0 if future_price > current_price else 0.0

        # IMPORTANT MODIFICATION: Ensure target column is at the end for MS mode
        if self.features == 'M' or self.features == 'MS':
            # Get all columns except date and target
            cols = list(df_raw.columns)

            # Remove date-related columns
            for col in ['date', 'date_readable', 'timestamp', 'close_time', 'split']:
                if col in cols:
                    try:
                        cols.remove(col)
                    except ValueError:
                        pass  # Column not in list

            # Remove target column from feature list
            if self.target in cols:
                cols.remove(self.target)

            # Create feature dataframe WITHOUT target
            df_data = df_raw[cols]
            active_df_data = active_data[cols]
            train_df_data = train_data[cols]

            # Now add target as the LAST column for MS mode
            if self.features == 'MS':
                print(f"Ensuring target column '{self.target}' is at the end for MS mode")
                # We'll append the target after scaling below

        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            active_df_data = active_data[[self.target]]
            train_df_data = train_data[[self.target]]

        # Before scaling, add checks and cleanups
        # Check for NaN or infinite values in raw data
        if np.isnan(df_data.values).any() or np.isinf(df_data.values).any():
            print("WARNING: Raw data contains NaN or infinite values")
            # Replace NaN/inf with zeros
            df_data = df_data.fillna(0)
            active_df_data = active_df_data.fillna(0)
            train_df_data = train_df_data.fillna(0)

        # Check for features with zero variance (will cause StandardScaler to produce NaNs)
        data_var = np.var(train_df_data.values, axis=0)
        zero_var_features = np.where(data_var == 0)[0]
        if len(zero_var_features) > 0:
            print(f"WARNING: Found {len(zero_var_features)} features with zero variance")
            # Identify zero variance columns by name
            zero_var_cols = [df_data.columns[i] for i in zero_var_features]
            print(f"Zero variance columns: {zero_var_cols[:10]}..." if len(
                zero_var_cols) > 10 else f"Zero variance columns: {zero_var_cols}")

            # Drop zero variance columns
            for col in zero_var_cols:
                if col in df_data.columns:
                    df_data = df_data.drop(columns=[col])
                if col in active_df_data.columns:
                    active_df_data = active_df_data.drop(columns=[col])
                if col in train_df_data.columns:
                    train_df_data = train_df_data.drop(columns=[col])

            print(f"After dropping zero variance features, remaining features: {df_data.shape[1]}")

        # Now try to scale with a more robust approach
        print(f"self.scale is: {self.scale}")
        if self.scale:
            try:
                self.scaler = StandardScaler()
                self.scaler.fit(train_df_data.values)
                data = self.scaler.transform(df_data.values)

                # Check for NaNs after scaling
                if np.isnan(data).any():
                    print("WARNING: StandardScaler produced NaN values, using manual scaling instead")
                    # Manual scaling as fallback
                    data_mean = np.nanmean(train_df_data.values, axis=0)
                    data_std = np.nanstd(train_df_data.values, axis=0)
                    # Handle zero std to avoid division by zero
                    data_std[data_std == 0] = 1.0
                    # Manual Z-score normalization
                    data = (df_data.values - data_mean) / data_std
            except Exception as e:
                print(f"ERROR: Failed to scale features: {e}")
                # Use raw data with simple normalization as last resort
                data = df_data.values
        else:
            data = df_data.values

        # ADD FEATURE NORMALIZATION VERIFICATION HERE
        # Verify scaled features are in a reasonable range
        print(
            f"Scaled features stats - min: {np.min(data):.4f}, max: {np.max(data):.4f}, mean: {np.mean(data):.4f}, std: {np.std(data):.4f}")

        # Check for any extreme values that could cause issues
        if np.max(np.abs(data)) > 100:
            print("WARNING: Very large scaled feature values detected!")
            # Clip extreme values
            data = np.clip(data, -10, 10)
            print(f"After clipping - min: {np.min(data):.4f}, max: {np.max(data):.4f}")

        # For MS mode, now add the target column at the end
        if self.features == 'MS':
            # Get the target column data
            target_data = df_raw[self.target].values.reshape(-1, 1)

            # For visualization, print some stats about target
            print(f"Target '{self.target}' stats - mean: {np.mean(target_data):.4f}, std: {np.std(target_data):.4f}")

            # Append target to scaled features
            data = np.concatenate([data, target_data], axis=1)

        # Create timestamp features
        if has_split_column:
            # For the new split method, get timestamps from active dataset
            df_stamp = active_data[['date']].reset_index(drop=True)
        else:
            # For traditional split method, get timestamps from the window
            df_stamp = df_raw[['date']][border1:border2]

        # Convert timestamps to datetime objects for feature extraction
        try:
            df_stamp['date'] = pd.to_datetime(df_stamp['date'], unit='s')

            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

        except Exception as e:
            print(f"Warning: Error creating timestamp features: {e}")
            # Create dummy time features as fallback
            data_stamp = np.zeros((len(df_stamp), 5))  # 5 time features (month, day, weekday, hour, minute)

        # Store data for the relevant window
        if has_split_column:
            # For the new split method, extract from the active dataset
            if self.features == 'MS':
                # For MS mode, we need to handle the concatenated data
                active_x = self.scaler.transform(active_df_data.values)
                active_target = active_data[self.target].values.reshape(-1, 1)
                self.data_x = np.concatenate([active_x, active_target], axis=1)
            else:
                self.data_x = self.scaler.transform(active_df_data.values)

            # Get corresponding binary labels for active dataset
            active_indices = active_data.index.tolist()
            self.data_y = binary_labels[active_indices]
        else:
            # For traditional split method, use the window
            self.data_x = data[border1:border2]
            self.data_y = binary_labels[border1:border2]

        self.data_stamp = data_stamp

        # Print training data info
        if self.set_type == 'train':
            print(f"train {len(self.data_x)}")
            if len(self.data_y) > 0:
                pos_examples = np.sum(self.data_y == 1.0)
                total_examples = len(self.data_y)
                pos_percentage = pos_examples / total_examples * 100
                print(f"Class distribution in training data:")
                print(f"Positive examples (price increases): {pos_percentage:.2f}%")
                print(f"Negative examples (price decreases or stays): {100 - pos_percentage:.2f}%")
                print(f"Pos_weight for BCEWithLogitsLoss: {(total_examples - pos_examples) / pos_examples:.4f}")
                print(f"Trading strategy: Shorting enabled")

    def __getitem__(self, index):
        # Your existing code to get sequences
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Get input sequence
        seq_x = self.data_x[s_begin:s_end]

        # Get target sequence
        seq_y = np.zeros((self.label_len + self.pred_len, 1))

        # Fill history and prediction parts
        if r_begin >= 0:
            seq_y[:self.label_len, 0] = self.data_y[r_begin:r_begin + self.label_len, 0]

        if r_begin + self.label_len < len(self.data_y):
            seq_y[self.label_len:, 0] = self.data_y[r_begin + self.label_len:r_end, 0]

        # Get timestamp features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Ensure seq_y is 2D [sequence_length, 1]
        if len(seq_y.shape) == 1:
            seq_y = seq_y.reshape(-1, 1)

        # Before returning data in __getitem__
        if np.isnan(seq_x).any() or np.isnan(seq_y).any():
            print(f"WARNING: NaN values detected in dataset at index {index}")
            # Replace NaNs with zeros
            seq_x = np.nan_to_num(seq_x, nan=0.0)
            seq_y = np.nan_to_num(seq_y, nan=0.0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
