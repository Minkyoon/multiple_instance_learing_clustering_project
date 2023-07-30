import shutil
import os

src_dir = "/home/minkyoon/2023_crohn_data/processed_data5"
dst_dir = "/home/minkyoon/2023_crohn_data/processed_data4"

# src_dir 안의 각 폴더에 대해
for folder_name in os.listdir(src_dir):
    src_folder_path = os.path.join(src_dir, folder_name)
    dst_folder_path = os.path.join(dst_dir, folder_name)

    # dst_dir에 동일한 이름의 폴더가 이미 있다면 삭제
    if os.path.exists(dst_folder_path):
        shutil.rmtree(dst_folder_path)

    # 폴더 복사 (src_dir의 폴더를 dst_dir로)
    shutil.copytree(src_folder_path, dst_folder_path)
