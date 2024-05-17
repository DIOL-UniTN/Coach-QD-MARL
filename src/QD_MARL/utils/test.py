def remove_previous_files(path):
    for file in os.listdir():
        remove(file)