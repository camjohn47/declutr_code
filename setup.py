from os import getcwd
from os.path import dirname, join, isdir

from glob import glob
import site
from pathlib import Path

import subprocess
import sys

get_dir_paths = lambda dir: glob(join(dir, "*"))
dir_has_python_init = lambda dir: join(dir, "__init__.py") in get_dir_paths(dir)

def get_subdir_paths():
    # The main_dir should point to declutr_code directory where subdirectories are placed.
    main_dir = getcwd()
    print(f"UPDATE: main dir = {main_dir}")
    sub_dirs = [path for path in get_dir_paths(main_dir) if isdir(path)]
    python_sub_dirs = filter(dir_has_python_init, sub_dirs)
    return python_sub_dirs

def write_sys_extension_line(extended_dir):
    line = f"sys.path.extend(['{extended_dir}'])"
    return line

def get_user_packages_path():
    site_packages_dir = site.getsitepackages()[0]
    user_packages_path = join(site_packages_dir, "usercustomize.py")
    return user_packages_path

def create_user_packages_script():
    user_packages_path = get_user_packages_path()
    user_packages_dir = dirname(user_packages_path)
    print(f"UPDATE: Creating user packages directory = {user_packages_dir}. (It may already exist.)")
    Path(user_packages_dir).mkdir(exist_ok=True, parents=True)
    sub_dirs = get_subdir_paths()
    extension_lines = [write_sys_extension_line(sub_dir) for sub_dir in sub_dirs]
    main_extension_line = write_sys_extension_line(getcwd())
    extension_lines.append(main_extension_line)
    print(f"UPDATE: Writing user packages script in {user_packages_path}. \n Extension lines = {extension_lines}")

    with open(user_packages_path, "w") as user_packages_file:
        user_packages_file.write(f"import sys \n")

        for line in extension_lines:
            user_packages_file.write(f"{line} \n")

def run_user_packages_script():
    user_packages_path = get_user_packages_path()
    print(f"UPDATE: Running user packages script in {user_packages_path}.")
    subprocess.run(["python3", user_packages_path])

create_user_packages_script()
run_user_packages_script()
print(f"UPDATE: Sys path = {sys.path}.")
