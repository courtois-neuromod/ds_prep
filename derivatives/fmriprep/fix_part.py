import os
import re
import glob
import shutil
import datalad.api

ds = datalad.api.Dataset("./")

for f in glob.glob("sub-*/ses-*/func/*part-mag*"):
    new_path = re.sub(r'(_run-[0-9]+)(.*)(_part-mag)', r'\1_part-mag\2', f)
    print(f, new_path)
    ds.repo.call_git(['mv', f, new_path])
