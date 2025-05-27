import os
import re


def count_files_by_year_in_name(
    base_dir: str,
    folder_name: str,
    start_year: int = 1999,
    end_year: int = 2025,
    fix_count: int = 5,
) -> None:

    """
    Counts files in `base_dir/folder_name` whose names contain
    each year from start_year to end_year anywhere in the filename.
    If the count != fix_count, prints which months are actually present
    and which are missing, in the format:
        2010: [1,2,3,4,...]      # present
        2010: [7,8]              # missing
    """
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_files = os.listdir(folder_path)
    wrong_years = {}

    # only match filenames like "YYYY-MM.csv"
    pattern = re.compile(r"(\d{4})-(\d{2})\.csv$")

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        # all files containing that year somewhere
        matching = [fn for fn in all_files if year_str in fn]
        cnt = len(matching)

        # if the count isn't what we expect, record which months are present
        if cnt != fix_count:
            months_present = set()
            for fn in matching:
                m = pattern.search(fn)
                if m and m.group(1) == year_str:
                    months_present.add(int(m.group(2)))
            wrong_years[year] = sorted(months_present)

    # --- 1) print present months ---
    print("Present months by year (count != {}):".format(fix_count))
    for year, months in wrong_years.items():
        print(f"    {year}: {months},")

    # --- 2) print missing months ---
    print("\nMissing months by year:")
    all_months = set(range(1, 13))
    #all_months =set([3,5,7,10, 12])
    for year, months in wrong_years.items():
        missing = sorted(all_months - set(months))
        if(missing != []):
            print(f"    {year}: {missing},")


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\ralph\PycharmProjects\Seasonal-Trading-in-Commodity-Markets\Complete Data"
    FOLDER = "CF_Historic_Data"

    count_files_by_year_in_name(BASE_DIR, FOLDER)
