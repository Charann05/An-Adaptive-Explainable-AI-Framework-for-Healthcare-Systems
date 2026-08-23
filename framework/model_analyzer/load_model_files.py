from pathlib import Path
import cli_display

SUPPORTED_EXTENSIONS = {
    ".py", ".ipynb", ".pth", ".pt", ".h5", 
    ".pkl", ".json", 
    ".csv", ".xls", ".xlsx", ".txt",
}

EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
}

def is_valid_project_folder(folder):
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in file.parts):
            continue
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
    return False

def collect_files(folder):
    files = []
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in file.parts):
            continue
        files.append(str(file.relative_to(folder)))
    return files

def get_model_folder():
    MODEL_FOLDER = Path.cwd()

    if is_valid_project_folder(MODEL_FOLDER):
        cli_display.print_status(f"Source project detected: {MODEL_FOLDER}")
        return MODEL_FOLDER

    print()
    cli_display.print_error(f"No recognizable model/project files found in: {MODEL_FOLDER}")

    while True:
        user_path = input("\nEnter the path to your model project: ").strip()

        if not user_path:
            print("Please enter a valid path.")
            continue

        MODEL_FOLDER = Path(user_path).expanduser().resolve()

        if not MODEL_FOLDER.exists():
            print(f"Path does not exist: {MODEL_FOLDER}")
            continue

        if not MODEL_FOLDER.is_dir():
            print(f"Path is not a directory: {MODEL_FOLDER}")
            continue

        if not is_valid_project_folder(MODEL_FOLDER):
            print(f"No recognizable model/project files found in: {MODEL_FOLDER}")
            continue

        cli_display.print_status(f"Source project detected: {MODEL_FOLDER}")
        return MODEL_FOLDER

def load_files():
    MODEL_FOLDER = get_model_folder()
    files = collect_files(MODEL_FOLDER)
    return MODEL_FOLDER, files