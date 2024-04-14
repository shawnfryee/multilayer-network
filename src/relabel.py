# Relabel incorrect files
import os


data_folder = os.path.join(os.path.dirname(os.getcwd()), "dataset/")

for i in range(839, 1470):
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")

    print(labels_folder + "/frame_" + str(i) + '.txt')

    f = open(f"{labels_folder}/frame_{str(i)}.txt", 'r')
    line = f.read()
    print(line)
    # f.write('left')
    # f.close()
