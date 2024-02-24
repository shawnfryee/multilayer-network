import os
import configparser

def make_output_directory():
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, "output")
    # Check if the directory exists
    if not os.path.exists(output_directory):
        # Create the directory if it doesn't exist
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created successfully.")
    else:
        print(f"Directory '{output_directory}' already exists.")

def main():
    make_output_directory()
    #get config parameters
    config_file = os.path.join(os.getcwd(), "config.ini")
    config = configparser.ConfigParser()
    config.read(config_file)
    #get dataset
    data_folder = os.path.join(os.getcwd(), "dataset/")


    print(data_folder)

    #setup training stuff

if __name__ == "__main__":
    main()