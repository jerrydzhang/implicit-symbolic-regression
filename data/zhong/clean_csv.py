import fileinput
from pathlib import Path

root_dir = Path(__file__).parent
data_dirs = [dir for dir in root_dir.iterdir() if dir.is_dir()]

for data_dir in data_dirs:
    base_data_path = data_dir / f"{data_dir.name}.csv"
    with fileinput.FileInput(base_data_path, inplace=True) as file:
        for line in file:
            print(line.replace(" ", ","), end="")
