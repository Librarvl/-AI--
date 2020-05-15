import csv
import time

file_addr = "../dataset/used_car_train_20200313/used_car_train_20200313.csv"

with open(file_addr, "r") as f:
    reader = csv.reader(f)

    for data in reader:
        print(data)
        time.sleep(1)
        # break