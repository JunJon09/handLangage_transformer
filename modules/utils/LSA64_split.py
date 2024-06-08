import os
import shutil

#LSA64のデータセットをトレーニング用とテスト用に分割
def DatasetSplit():
    folder_path = "../LSA64/all/"
    files_and_directories = os.listdir(folder_path)

# ファイル名のみを取得（ディレクトリを除外）
    file_names = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]

    sorted_file_names = sorted(file_names)
    class_count = -1 #クラス数
    number = 0
    #shutil.copy("../LSA64/all/008_010_003.mp4", "../test/01.mp4")

    train_list = []
    test_list = []
    for i, file_name in enumerate(sorted_file_names):
        if i % 50 == 0:
            class_count += 1
            print(i, class_count)
            path = "../data/pre/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)
            path = "../data/test/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)
            path = "../data/train/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)

        basis_copy_path = "../LSA64/all/" + file_name
        x = i % 50
        print(x)
        if 0 <= x and x <35: #19 pre
            next_path = "../data/pre/" + str(class_count).zfill(3) + "/" + file_name
        elif 35 <= x and x < 40: #train
            next_path = "../data/train/" + str(class_count).zfill(3) + "/" + file_name
        elif 40<= x and x <50:#test
            next_path = "../data/test/" + str(class_count).zfill(3) + "/"  + file_name
            
        shutil.copy(basis_copy_path, next_path)


DatasetSplit()