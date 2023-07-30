import shutil

source_folder = "/home/minkyoon/2023_crohn_data/processed_data4"
destination_folder = "/home/minkyoon/2023_crohn_data/processed_original"

print('시작')
shutil.copytree(source_folder, destination_folder)
print('끝~')