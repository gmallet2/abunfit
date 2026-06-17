import json
from pathlib import Path
from abunfit import Tools
periodic_table = Tools.build_periodic_table()
data = {}

folder = Path("data/models/SNIa")

for txt_file in folder.glob("*.txt"):
    print(txt_file)
    txt_name = str(txt_file).split("/")[-1][:-4]
    print(txt_name)
    with open(txt_file, "r") as f:
        data_file = {"Mass":None}
        try :
            for i, line in enumerate(f, start=1):
                new_key=""
                data = [float(x) for x in line.split()]
                if len(data) == 3 :
                    Z = int(data[0])
                    A = int(data[1])
                    for i in list(periodic_table.keys()) : 
                        if periodic_table[i]['Z'] == Z :
                            name_element = i
                            new_key = name_element+"_"+str(A)
                    data_file[new_key] = data[2]
        except(ValueError) : pass
        print(data_file)

    with open("data/models/SNIa/"+txt_name+".json", "w") as f:
        json.dump(data_file, f, separators=(",", ":"))


