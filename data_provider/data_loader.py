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
        """
                Dataset for cryptocurrency price forecasting with PCA components

                Args:
                    root_path: root path of the data file
                    flag: 'train', 'test' or 'val'
                    size: [seq_len, label_len, pred_len]
                    features: 'M', 'MS' or 'S'
                    data_path: data file
                    target: target column (e.g., 'close')
                    scale: whether to scale the data
                    timeenc: time encoding method
                    freq: time frequency
        """
        # size is: [96, 1, 1]
        if size == None:
            self.seq_len = 96  # 24 hours (96 * 15 min)
            self.label_len = 48
            self.pred_len = 1  # Predict 1 step ahead
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']

        # Store parameters
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        # Only print detailed information for the training dataset to avoid redundancy
        self.is_train = (flag == 'train')

        self.__read_data__()

        # Print summary for training dataset only
        if self.is_train:
            self.__print_summary__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Store dataset properties for summary
        self.total_rows = len(df_raw)
        self.has_split_column = 'split' in df_raw.columns

        # Save a copy of the raw dataframe with split column intact for debugging
        if self.flag == 'test' and self.has_split_column:
            df_raw_with_split = df_raw.copy()

        # Process data splits
        if self.has_split_column:
            # Use the split column to filter data
            train_data = df_raw[df_raw['split'] == 'train']
            val_data = df_raw[df_raw['split'] == 'val']
            test_data = df_raw[df_raw['split'] == 'test']

            # Store split information
            self.split_info = {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            }

            # Remove the split column before feature processing
            train_data = train_data.drop(columns=['split'])
            val_data = val_data.drop(columns=['split'])
            test_data = test_data.drop(columns=['split'])

            # Set the appropriate dataset based on flag
            if self.flag == 'train':
                active_data = train_data
            elif self.flag == 'val':
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
            set_type_idx = type_map[self.flag]

            border1 = border1s[set_type_idx]
            border2 = border2s[set_type_idx]

            # Set active data
            active_data = df_raw
            train_data = df_raw.iloc[:num_train]

        # Create binary labels for price change prediction (1 if price goes up, 0 if down/same)
        close_prices = df_raw[self.target].values
        binary_labels = np.zeros((len(close_prices), 1))

        # For each potential sequence starting point
        for i in range(len(close_prices) - self.seq_len - self.pred_len):
            # Calculate prediction point index
            pred_idx = i + self.seq_len

            # Get prices at prediction point and future point
            pred_price = close_prices[pred_idx]
            future_price = close_prices[pred_idx + self.pred_len]

            # Set label based on future price direction from prediction point
            binary_labels[i, 0] = 1.0 if future_price > pred_price else 0.0

        # Store active indices before dropping the split column
        active_indices = active_data.index.tolist()
        self.active_indices = active_indices  # Save as class attribute

        # Add this after binary_labels creation in __read_data__
        self.sample_metadata = []
        for i in range(len(self.active_indices)):
            orig_idx = self.active_indices[i]
            pred_idx = orig_idx + self.seq_len

            if pred_idx < len(close_prices) and pred_idx + self.pred_len < len(close_prices):
                pred_price = close_prices[pred_idx]
                future_price = close_prices[pred_idx + self.pred_len]
                label = binary_labels[orig_idx, 0]  # Get the label that was calculated

                self.sample_metadata.append({
                    'sample_idx': i,
                    'orig_idx': orig_idx,
                    'pred_idx': pred_idx,
                    'pred_price': pred_price,
                    'future_price': future_price,
                    'label': label,
                    'timestamp': df_raw['date'].iloc[pred_idx] if 'date' in df_raw.columns else None
                })

        # Create a mapping from sequence indices to original indices
        self.sequence_indices = {}
        for i in range(len(self.active_indices) - self.seq_len - self.pred_len + 1):
            # Skip if we don't have enough data for this sequence
            if i >= len(self.active_indices) or i + self.seq_len >= len(self.active_indices):
                continue

            orig_start_idx = self.active_indices[i]
            pred_idx = orig_start_idx + self.seq_len
            future_idx = pred_idx + self.pred_len

            # Skip if we'd go beyond the end of the original data
            if future_idx >= len(close_prices):
                continue

            # Calculate price at prediction point and future point
            pred_price = close_prices[pred_idx]
            future_price = close_prices[future_idx]

            # Calculate label directly
            calculated_label = 1.0 if future_price > pred_price else 0.0

            # Store all relevant information
            self.sequence_indices[i] = {
                'orig_start_idx': orig_start_idx,
                'pred_idx': pred_idx,
                'future_idx': future_idx,
                'pred_price': pred_price,
                'future_price': future_price,
                'price_change': (future_price - pred_price) / pred_price * 100.0,
                'label': calculated_label
            }

        # Save original active_indices for integrity checking
        self._original_active_indices = self.active_indices.copy() if hasattr(self, 'active_indices') else None

        if 'split' in df_raw.columns:
            df_raw = df_raw.drop(columns=['split'])

        # Calculate price movement statistics
        self.price_stats = {
            'positive': np.sum(binary_labels),
            'total': len(binary_labels) - self.pred_len,
        }

        # Feature columns (all PCA components + direction; excludes date and close)
        if self.features == 'M' or self.features == 'MS':
            cols = list(df_raw.columns)
            # Always remove close price from features
            cols.remove(self.target)
            # Always remove date (processed separately)
            cols.remove('date')
            print(f"{self.flag}: Removing date column")

            # Save feature list for summary
            self.feature_list = cols

            # Extract appropriate data
            df_data = df_raw[cols]
            active_df_data = active_data[cols]
            train_df_data = train_data[cols]

        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            active_df_data = active_data[[self.target]]
            train_df_data = train_data[[self.target]]

        else:
            raise ValueError(f"Invalid features argument: {self.features}")

        # Scale the features
        if self.scale:
            # Always fit scaler on training data only
            self.scaler.fit(train_df_data.values)
            # Store scaling stats
            self.scaling_stats = {
                'mean_range': [np.min(self.scaler.mean_), np.max(self.scaler.mean_)],
                'std_range': [np.min(np.sqrt(self.scaler.var_)), np.max(np.sqrt(self.scaler.var_))]
            }
            # Transform all data
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process timestamps
        if self.has_split_column:
            # For the new split method, get timestamps from active dataset
            df_stamp = active_data[['date']].reset_index(drop=True)
        else:
            # For traditional split method, get timestamps from the window
            df_stamp = df_raw[['date']][border1:border2]

        # Convert to datetime
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # Create time features
        if self.timeenc == 0:
            # Manual time encoding
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
            self.time_encoding = 'manual'
        elif self.timeenc == 1:
            # Automatic time features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            print(f"the shape of data_stamp is: {data_stamp.shape}")
            self.time_encoding = 'auto'
        else:
            raise ValueError(f"Invalid timeenc argument: {self.timeenc}")

        # Store processed data
        if self.has_split_column:
            # For split based method, extract from the active dataset
            self.data_x = self.scaler.transform(active_df_data.values)
            # Get corresponding binary labels for active dataset
            active_indices = active_data.index.tolist()
            self.data_y = binary_labels[active_indices]
        else:
            # For traditional split method, use the window
            self.data_x = data[border1:border2]
            self.data_y = binary_labels[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """Get a single sample (input sequence, target sequence, and time features)"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Get input sequence
        seq_x = self.data_x[s_begin:s_end]

        # Get target sequence - this is where we ensure correct label
        seq_y = np.zeros((self.label_len + self.pred_len, 1))

        # Fill history part of target
        if r_begin >= 0:
            seq_y[:self.label_len, 0] = self.data_y[r_begin:r_begin + self.label_len, 0]

        # Fill prediction part of target
        if r_begin + self.label_len < len(self.data_y):
            # For test set, use our sequence mapping to ensure correct label
            if self.flag == 'test' and hasattr(self, 'sequence_indices') and index in self.sequence_indices:
                # Use the stored label for the prediction part
                seq_y[self.label_len:, 0] = self.sequence_indices[index]['label']
            else:
                # Standard behavior for train/val
                seq_y[self.label_len:, 0] = self.data_y[r_begin + self.label_len:r_end, 0]

        # Get timestamp features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Ensure seq_y is 2D [sequence_length, 1]
        if len(seq_y.shape) == 1:
            seq_y = seq_y.reshape(-1, 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """Return the effective length of the dataset after accounting for sequence windows"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse transform scaled data back to original scale"""
        return self.scaler.inverse_transform(data)

    def verify_indices_integrity(self):
        """Verify that active_indices has not been modified since initialization"""
        if not hasattr(self, '_original_active_indices'):
            print("WARNING: No original active indices saved for verification")
            return False

        if not hasattr(self, 'active_indices'):
            print("WARNING: No active_indices attribute found")
            return False

        if len(self._original_active_indices) != len(self.active_indices):
            print(
                f"ERROR: active_indices length changed! Original: {len(self._original_active_indices)}, Current: {len(self.active_indices)}")
            return False

        # Check first 5 and last 5 elements
        check_first = all(
            self._original_active_indices[i] == self.active_indices[i] for i in range(min(5, len(self.active_indices))))
        check_last = all(self._original_active_indices[-(i + 1)] == self.active_indices[-(i + 1)] for i in
                         range(min(5, len(self.active_indices))))

        if not check_first or not check_last:
            print("ERROR: active_indices values have changed!")
            print(f"Original first 5: {self._original_active_indices[:5]}")
            print(f"Current first 5: {self.active_indices[:5]}")
            print(f"Original last 5: {self._original_active_indices[-5:]}")
            print(f"Current last 5: {self.active_indices[-5:]}")
            return False

        return True

    def __print_summary__(self):
        """Print a concise summary of the dataset (only called for training dataset)"""
        print(f"\n{'=' * 20} DATASET SUMMARY {'=' * 20}")

        # Basic dataset info
        print(f"Dataset: {self.data_path}")
        print(f"Total rows: {self.total_rows}")

        # Split information
        print(f"\nSplit distribution:")
        for split, count in self.split_info.items():
            print(f"  - {split}: {count} samples ({count / self.total_rows:.2%})")

        # Feature configuration
        print(f"\nFeature configuration:")
        print(f"  - Total features: {len(self.feature_list)}")

        # Price movement statistics
        positive_ratio = self.price_stats['positive'] / self.price_stats['total']
        print(f"\nPrice movement statistics:")
        print(f"  - Increases: {self.price_stats['positive']} ({positive_ratio:.2%})")
        print(
            f"  - Decreases/No change: {self.price_stats['total'] - self.price_stats['positive']} ({1 - positive_ratio:.2%})")

        # Feature scaling info
        if self.scaling_stats:
            print(f"\nFeature scaling statistics:")
            print(
                f"  - Mean range: [{self.scaling_stats['mean_range'][0]:.4f}, {self.scaling_stats['mean_range'][1]:.4f}]")
            print(
                f"  - Std range: [{self.scaling_stats['std_range'][0]:.4f}, {self.scaling_stats['std_range'][1]:.4f}]")

        # Model parameters vs. data dimensions
        print(f"\nModel configuration:")
        print(f"  - Sequence length: {self.seq_len}")
        print(f"  - Label length: {self.label_len}")
        print(f"  - Prediction length: {self.pred_len}")
        print(f"  - Time encoding: {self.time_encoding} ({self.data_stamp.shape[1]} features)")

        # Effective samples after windowing
        effective_samples = len(self.data_x) - self.seq_len - self.pred_len + 1
        print(f"  - Effective samples: {effective_samples}")


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
