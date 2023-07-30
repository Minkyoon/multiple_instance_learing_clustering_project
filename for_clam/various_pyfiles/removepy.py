import os
import glob

# 지울 파일들이 있는 디렉토리 경로
directory = "/home/minkyoon/crohn/csv/clam/relapse"

# 해당 디렉토리 내의 모든 .pt 파일들의 경로를 리스트로 가져옴
pt_files = glob.glob(os.path.join(directory, "*.pt"))

# 각 .pt 파일을 하나씩 지움
for pt_file in pt_files:
    os.remove(pt_file)

print("All .pt files have been deleted.")



import os

# CSV 파일이 있는 디렉토리 경로
csv_directory = "/home/minkyoon/crohn/csv/clam/relapse"

# CSV 파일들의 이름 (확장자 제외)을 리스트로 가져오기
csv_files = [os.path.splitext(filename)[0] for filename in os.listdir(csv_directory) if filename.endswith(".csv")]

# 비교할 accession_number 리스트
#미니맵오류
accession_numbers = ["4", "22", "23", "33", "39", "80", "107", "110", "158", "166", "167", "187", "195", "210", "279", "281", "284", "293", "309", "335", "340", "385", "390", "391", "424", "425", "430", "461", "466", "473", "511", "519", "526", "533", "553", "556", "584", "601", "620", "658", "681", "691", "723", "773", "807", "828", "840", "949", "952", "953", "980", "994", "1008", "1028"]
# 비교할 새로운 accession_number 리스트
#회색오류
new_accession_numbers1 = ["1", "45", "59", "79", "119", "135", "142", "178", "184", "190", "206", "230", "236", "268", "283", "341", "436", "438", "444", "464", "476", "565", "603", "613", "630", "635", "662", "743", "757", "775", "792", "795", "820", "825", "844", "848", "851", "853", "878", "884", "913", "926", "927", "931", "936", "955", "956", "957", "958", "959", "964", "966", "975", "992"]
#패킹오류
new_accession_numbers2 = ["233", "235", "237", "241", "256", "317", "332", "323", "362", "367", "380", "401", "410", "418", "421", "427", "432", "434", "443", "449", "450", "458", "460", "462", "490", "491", "492", "507", "527", "530", "567", "570", "577", "578", "580", "582", "591", "608", "626", "639", "644", "687", "699", "707", "708", "718", "719", "721", "748", "759", "756", "786", "799", "800", "806", "821", "822", "832", "870", "868", "920", "923", "930", "932", "945", "988", "997", "999", "1000", "1004", "1005", "1006", "1017", "1020", "1023", "1027", "1037", "1038", "1040", "1110", "1122"]
#아래글자오류
new_accession_numbers3 = ["630", "757", "884", "913", "927", "933", "966"]
#기타오류
new_accession_numbers4 = ["73", "83", "225", "377", "588", "596", "617", "627", "629", "647", "747", "874", "1120", "1165", "1306", "1342"]

# 겹치는 accession_number 찾기
common_numbers1 = list(set(csv_files) & set(new_accession_numbers1))
common_numbers2 = list(set(csv_files) & set(new_accession_numbers2))
common_numbers3 = list(set(csv_files) & set(new_accession_numbers3))
common_numbers4 = list(set(csv_files) & set(new_accession_numbers4))
common_numbers = list(set(csv_files) & set(accession_numbers))

print("Common accession numbers are:", common_numbers)
print("Common accession numbers with the first list are:", common_numbers1)
print("Common accession numbers with the second list are:", common_numbers2)
print("Common accession numbers with the first list are:", common_numbers3)
print("Common accession numbers with the second list are:", common_numbers4)
