def get_files_to_keep(files, msg):
    files_to_keep = []
    for file in files:
        name = file["name"]
        if f"[[{name}]]" in msg:
            files_to_keep.append(file)
    return files_to_keep
