import bz2
import json
import tinydb

FILE_NAME = "RC_2005-12"

db = tinydb.TinyDB("{}.json".format(FILE_NAME))

insertions = 0

with bz2.open("../{}.bz2".format(FILE_NAME)) as raw_data:
    for lines in raw_data:
        db.insert(json.loads(lines))
        insertions += 1
        if insertions % 50 == 0:
            print("completed {} insertions".format(insertions))

print("Success!\nCompleted {} insertions!".format(insertions))
db.close()
