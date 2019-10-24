import csv
import pandas as pd

dfOdds = pd.DataFrame(columns=("Id", "Company", "Wins", "Draws", "Losses"))

f = open("./data/ENG_PR/odds.csv", "r")
line = f.readline()

i = 0
while line:
    line = f.readline()
    if (not line.startswith("oddsData[O")):
        print("invalid data:",line)
        continue;
    # print(line)

    line = line.replace("\n", "")
    arrLine = line.split("=")
    gid = arrLine[0][11:18]
    arrGOdds = arrLine[1].split("],[")
    for godds in arrGOdds:
        odds = godds.split(",")
        dfOdds.loc[i] = [gid, odds[0], odds[1], odds[2], odds[3]]
        i += 1


    # if(i > 20):
    #     break

f.close()

print(dfOdds)
dfOdds.to_csv('./data/ENG_PR/odds_format.csv', index=False)
# strTest = "oddsData[O_1552513]=432,17,7.4,1.1],[16,25.75,11.5,1.08],[82,29,10,1.1],[158,17,9,1.11],[173,16.81,8.96,1.13],[474,18,7,1.11],[110,18,9.5,1.11],[60,22.97,10,1.1],[90,22,7.8,1.12],[281,21,9,1.1],[18,23,7.2,1.09],[81,29,11,1.07],[88,29,10,1.1],[4,27,10,1.13],[70,26,11.75,1.12],[370,15,6.75,1.06],[422,17,9,1.11],[115,34,9,1.1],[255,17,9,1.11],[545,16,9.6,1.09],[659,16,9.6,1.09],[104,19,9.4,1.12],[80,17,9.5,1.06],[649,17,7.5,1.14],[450,16.5,8.7,1.11],[177,19.56,10.96,1.12],[499,16,9.6,1.09],[71,18,9,1.12],[976,19,11,1.11],[97,17.76,10.29,1.09"
# a = strTest.split("=")
# print(a[0].startswith("oddsData[O"))
# print(a[0][11:18])
# rate = a[1].split("],[")
# for arate in rate:
#     print(arate.split(","))