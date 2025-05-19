import os
import re

def count_files_by_year_in_name(
    base_dir: str,
    folder_name: str,
    start_year: int = 1999,
    end_year: int = 2025,
    fix_count: int = 5
):
    """
    Counts files in `base_dir/folder_name` whose names contain
    each year from start_year to end_year anywhere in the filename.
    If the count != fix_count, prints which months are actually present
    in the format:
        2008: [3, 5, 12],
        2000: [1, 2, 7],
        2025: [6, 9, 12],
    """
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_files = os.listdir(folder_path)
    wrong_years = {}

    pattern = re.compile(r"(\d{4})-(\d{2})\.csv$")

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        matching = [fn for fn in all_files if year_str in fn]
        cnt = len(matching)

        if cnt != fix_count:
            # extract months via regex YYYY-MM.csv
            months = set()
            for fn in matching:
                m = pattern.search(fn)
                if m and m.group(1) == year_str:
                    months.add(int(m.group(2)))
            wrong_years[year] = sorted(months)

    # Print in the requested format
    for year, months in wrong_years.items():
        print(f"        {year}: {months},")

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\ralph\PycharmProjects\Seasonal-Trading-in-Commodity-Markets\Complete Data"
    FOLDER   = "CC_Historic_Data"

    count_files_by_year_in_name(BASE_DIR, FOLDER)
