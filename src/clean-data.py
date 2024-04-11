import os


def reduce_files_in_directory(directory_path, keep_every=10):
    # List all files in the directory
    files = os.listdir(directory_path)
    files.sort()

    counter = 1

    for file in files:
        file_path = os.path.join(directory_path, file)

        if counter % keep_every != 0:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"Keeping {file_path}")
        counter += 1


def main():
    dataset_dir = os.getcwd()

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        reduce_files_in_directory(images_dir)
        reduce_files_in_directory(labels_dir)
    else:
        print("Both 'images' and 'labels' directories must exist in the dataset directory.")


if __name__ == "__main__":
    main()