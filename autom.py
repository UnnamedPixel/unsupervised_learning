import os

def list_files_with_specific_extension(folder_path, specific_extension):
    files_with_extension = []
    for file in os.listdir(folder_path):
        _, extension = os.path.splitext(file)
        if extension == f".{specific_extension}":
            files_with_extension.append(file)
    return files_with_extension

def group_files_by_name(folder_path):
    file_groups = {}
    for file in os.listdir(folder_path):
        filename, extension = os.path.splitext(file)
        base_name = filename.rstrip('1234567890')  # Retire les chiffres de la fin du nom
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)
    
    return list(file_groups.values())

# Utilisation de la fonction
dossier = os.path.dirname(__file__).replace("\\", "/") + "/artificialificial/"  # Remplacez ceci par le chemin de votre dossier
all_files = group_files_by_name(dossier)


