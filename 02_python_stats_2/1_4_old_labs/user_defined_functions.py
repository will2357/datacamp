import os

files = ["base.py"]

for i in files:
    exec(open(os.path.abspath(i)).read())



