# INSTALL

pip install -r requirements.txt

# EXTRACT HCPTRT TASK

```
usage: extract_hcptrt.py [-h] [--extract_eprime] [-f] [-v]
                         in_file in_task out_file

    Convert txt data from eprime to tsv using convert_eprime.
    git@github.com:tsalo/convert-eprime.git

    HCPTRT tasks
    https://github.com/hbp-brain-charting/public_protocols

positional arguments:
  in_file           Task output (.txt) to convert.
  in_task           Config JSON file defining the task you want to convert.
  out_file          output tsv file.

optional arguments:
  -h, --help        show this help message and exit
  --extract_eprime  Extract eprime file into raw tsv file.
                    It helps to create a config file.
  -f                Force overwriting of the output files.
  -v                If set, produces verbose output.
```

# WRITE YOUR OWN CONFIG FILE
