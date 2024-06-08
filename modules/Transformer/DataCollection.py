import os
import pandas as pd
import Transformer_config as config
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F


class DataCollection():
    def __init__(self) -> None:
        self.train = []
        self.var = []
        self.test = []

    #全てのcsvファイルを読み込む
    def read_csv(self) -> list:
        folder_path = config.LSA64_folder_path
        data_csvs = []
        labels = []

        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        sorted_csv_files = sorted(csv_files)
        for file in sorted_csv_files:
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, na_values=[''])
            # # データが空の場合、Noneに置き換え
            data = data.where(pd.notnull(data), 0)
            print(f'Processing file: {file}')
            labels.append([float(file[:3])])
            
            data_csvs.append(data)
        return data_csvs, labels

    #不要のデータ列を削除
    def remove_values(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_remove = config.pose_columns_to_remove + [col for col in df.columns if '_z' in col]
        remove_data = df.drop(columns=columns_to_remove)
        return remove_data


    #体は鼻を基準とし、手の甲を基準に変換する
    def convert_to_relative_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        relative_df = df.copy()

        def get_reference_values(prefix):
            return df[f"{prefix}_x"], df[f"{prefix}_y"]

        pose_reference_x, pose_reference_y = get_reference_values(config.pose_stating_point)
        left_reference_x, left_reference_y = get_reference_values(config.left_hand_stating_point)
        right_reference_x, right_reference_y = get_reference_values(config.right_hand_stating_point)
        for col in df.columns:
            if 'pose' in col:
                if col.endswith('_x'):
                    relative_df[col] = df[col] - pose_reference_x
                elif col.endswith('_y'):
                    relative_df[col] = df[col] - pose_reference_y
            elif 'left' in col:
                if col.endswith('_x'):
                    relative_df[col] = df[col] - left_reference_x
                elif col.endswith('_y'):
                    relative_df[col] = df[col] - left_reference_y
            elif 'right' in col:
                if col.endswith('_x'):
                    relative_df[col] = df[col] - right_reference_x
                elif col.endswith('_y'):
                    relative_df[col] = df[col] - right_reference_y

        return relative_df
    
    #パディングとマスク処理
    def padded_mask(self, skeleton_datas, max_len):
        padded_skeleton =[]
        mask_skeleton = []
        for skeleton_data in skeleton_datas:
            pad_len = max_len - skeleton_data.shape[0]
            skeleton_data = torch.FloatTensor(skeleton_data.values)
            padded_seq = F.pad(skeleton_data, (0, 0, 0, pad_len), value=0)
            padded_skeleton.append(padded_seq)
            mask = torch.ones(max_len, dtype=torch.bool)
            mask[skeleton_data.shape[0]:] = False
            mask_skeleton.append(mask)
        padded_skeleton = torch.stack(padded_skeleton)
        mask_skeleton = torch.stack(mask_skeleton)
        return padded_skeleton, mask_skeleton

    #事前データ, 検証データ, テストデータをセットする
    def get_dataset(self) -> None:
        data_csvs, labels = self.read_csv()
        skeleton_datas = []
        max_len = -1
        for data_csv in data_csvs:
            remove_data = self.remove_values(data_csv)
            relative_data = self.convert_to_relative_coordinates(remove_data)
            if max_len < relative_data.shape[0]:
                max_len = relative_data.shape[0]
            skeleton_datas.append(relative_data)
        print("*"* 100)
       

        skeleton_train, temp_data, labels_train, labels_temp = train_test_split(skeleton_datas, labels, test_size=0.3, random_state=42)
        skeleton_val, skeleton_test, labels_val, labels_test = train_test_split(temp_data, labels_temp, test_size=2/3, random_state=42)
        print(len(skeleton_datas), len(skeleton_train), len(skeleton_val), len(skeleton_test))
        del temp_data, skeleton_datas, labels_temp, labels
        labels_train = torch.FloatTensor(labels_train)
        labels_val = torch.FloatTensor(labels_val)
        labels_test = torch.FloatTensor(labels_test)
        skeleton_train, mask_train = self.padded_mask(skeleton_train, max_len)
        skeleton_val, mask_val = self.padded_mask(skeleton_val, max_len)
        skeleton_test, mask_test = self.padded_mask(skeleton_test, max_len)
        return skeleton_train, skeleton_val, skeleton_test, labels_train, labels_val, labels_test, mask_train, mask_val, mask_test
