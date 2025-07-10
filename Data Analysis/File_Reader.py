from pathlib import Path
import pandas as pd

def choose(prompt, options):
    for i, opt in enumerate(options):
        print(f"{i}: {opt}")
    idx = int(input(f"{prompt} [0–{len(options)-1}]: "))
    return options[idx]

def main():
    # this file: …/Data Analysis/File_Reader.py
    here: Path = Path(__file__).resolve().parent

    # project root is one level up
    project_root: Path = here.parent

    # now point at your “Complete Data” folder
    data_root: Path = project_root / "Complete Data"

    # 1) list subfolders
    subfolders = [p for p in data_root.iterdir() if p.is_dir()]
    print("Choose a subfolder:")
    folder = choose("Folder index", subfolders)

    # 2) list CSVs inside it
    csv_files = [p for p in folder.iterdir() if p.suffix.lower() == '.csv']
    print(f"Choose a file in '{folder.name}':")
    file = choose("File index", csv_files)

    # 3) read it
    print(f"\nLoading {file} …")
    df = pd.read_csv(file)

    print(df.head())

if __name__ == "__main__":
    main()
