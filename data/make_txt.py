import os

file_name = os.listdir('img/')
file = open('train_img.txt', 'w')
for item in file_name:
    file.write('data/train_data/img/' + item + '\n')
file.close()
