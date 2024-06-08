LSA64_folder_path = '../../csv/LSA64'
pose_columns_list = ['pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6', 'pose_7', 'pose_8', 'pose_9', 'pose_10', 'pose_25', 'pose_26', 'pose_27', 'pose_28', 'pose_29', 'pose_30', 'pose_31', 'pose_32']
pose_columns_to_remove = [suffix for text in pose_columns_list for suffix in [text + "_x", text + "_y"]]
pose_stating_point = "pose_0"
left_hand_stating_point = "left_hand_0"
right_hand_stating_point = "right_hand_0"

input_dim = 115 #入力の次元(カラム)
model_dim = 256  # モデル次元
num_heads = 8    # アテンションヘッド数
num_layers = 6  # Transformer層数
dropout = 0.3    # ドロップアウト率
num_classes = 64 #クラス数
max_len = 242
num_epochs= 3
batch_size = 4

restore_model_path = "../../models/transformer_model.pth"
