import bz2
import json
import tinydb
import argparse
import pdb

def main(data_name, db_name):
    print(data_name)
    print(db_name)
    db = tinydb.TinyDB(db_name)

    insertions = 0

    with bz2.open(data_name) as raw_data:
        for lines in raw_data:
            db.insert(json.loads(lines))
            insertions += 1
            if insertions % 50 == 0:
                print("completed {} insertions".format(insertions))

    print("Success!\nCompleted {} insertions!".format(insertions))
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest a b-zipped json file into a tinydb database')
    parser.add_argument("db_name", help="What you'd like the new database to be called")
    parser.add_argument("bz_name", help="Location of existing bz2 datafile")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args.bz_name, args.db_name)
