import json
import bz2
from pprint import pprint 

file_location="../RC_2010-01.bz2"

with bz2.open(file_location, mode='rt') as data:
    pprint(json.loads(data.readline()))

