#coding=utf-8
import random
import traceback

import requests
import time
import datetime
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


headers = {"Host": "zq.win007.com",
           "Connection": "keep-alive",
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36",
           "Accept": "*/*",
           "Referer": "http://zq.win007.com/cn/League/2018-2019/36.html",
           "Accept-Encoding": "gzip, deflate",
           "Accept-Language": "zh-CN,zh;q=0.9"}


cookies = dict(UM_distinctid="16dec6dd1ec995-0009fac8777f12-b363e65-1fa400-16dec6dd1edae3",
               bfWin007FirstMatchTime="2019,9,22,08,00,00", win007BfCookie="2^0^1^1^1^1^1^0^0^0^0^0^1^2^1^1^0^1^1^0",
               detailCookie="1^0^1",
               CNZZDATA1261430177="2050677833-1571624485-http%253A%252F%252Fwww.win007.com%252F%7C1571905339")
# 获取当前时间yyyyMMddHH
def getNowH():
    return datetime.datetime.now().strftime('%Y%m%d%H')

# 获取当前时间戳
def getNowTimeStamp():
    return time.time()

# ===========================赛程数据处理===========================
def crwalMatch():
    # 赛程数据Url
    def getMatchResultUrl(year):
        return str("http://zq.win007.com/jsData/matchResult/%s-%s/s36.js?version=%s" % (year, year + 1, getNowH()))

    colResult = ["Id", "Year", "Round", "League", "Date", "Home", "Visiters", "FinalScore", "HalfScore","Wins","Draws","Losses"]

    # 解析每轮数据
    def analysisRoundData(year, line):
        line = line.replace('"', "").replace("'","")
        arrLine = line.split("=")
        roundNum = arrLine[0][5:7].replace("]","") # 轮数
        arrRoundData = arrLine[1][2:].replace("]];","").split("],[")
        dfReturn = pd.DataFrame(columns=colResult)
        for roundData in arrRoundData:
            round = roundData.split(",")
            # 解析胜平负
            arrScore = round[6].split("-")

            wins=draws=losses = 0
            if (int(arrScore[0]) > int(arrScore[1])):
                # 胜
                wins = 1
            elif (int(arrScore[0]) == int(arrScore[1])):
                # 平
                draws = 1
            else:
                # 负
                losses = 1
            dfReturn.loc[dfReturn.shape[0] + 1] = [round[0], year, roundNum, round[1], round[3], round[4], round[5],
                                                   round[6], round[7], wins, draws, losses]
        return dfReturn

    # 抓取结果数据
    i = 2018
    while i > 2004:
        try:
            url = getMatchResultUrl(i)
            print("Get Result Data from Url[%s]" % (url))
            headers["Referer"] = "http://zq.win007.com/cn/League/%s-%s/36.html" % (i, i + 1)
            print(headers)
            resp = requests.session().get(url, headers=headers, cookies=cookies, timeout=5)
            print(resp.text)
            arrData = resp.text.split("\n")
            dfResult = pd.DataFrame(columns=colResult)  # 结果数据
            for j in range(2, 40, 1):
                # print(arrData[j])
                dfYearResult = analysisRoundData(i, arrData[j])
                dfResult = pd.concat([dfResult, dfYearResult], ignore_index=True)

            print(dfResult.head())
            # 写入文件
            # dfResult.to_csv('./data/ENG_PR/result.csv', index=False, mode='a', header=False)
            i -= 1
        except Exception as e:
            print('traceback.format_exc():\n%s' % traceback.format_exc())

        time.sleep(random.randint(3, 5))
# ===========================赛程数据处理===========================

# ===========================赔率数据处理===========================
def crwalOdds():
    # 赛程数据Url
    def getOddsUrl(year,round):
        return str("http://zq.win007.com/League/LeagueOddsAjax?sclassId=36&subSclassId=0&matchSeason=%s-%s&round=%s&flesh=%s" % (year, year + 1,round, getNowTimeStamp()))

    colOdds = ["Id", "Type", "Company", "WinsOdds", "DrawsOdds", "LossesOdds"]

    # 解析赔率数据
    def analysisOddsData(roundData):
        roundData = roundData.replace('"', "").replace("'", "")
        arrLine = roundData.split(";")
        dfReturn = pd.DataFrame(columns=colOdds)
        for line in arrLine:
            if (not line.startswith("oddsData")):
                break
            arrOdds = line.split("=")
            type = arrOdds[0][9:10]
            gid = arrOdds[0][11:18].replace("]","")
            arrGOdds = arrOdds[1].replace("]]", "").replace("[[", "").split("],[")
            for godds in arrGOdds:
                odds = godds.split(",")
                dfReturn.loc[dfReturn.shape[0] + 1] = [gid, type, odds[0], odds[1], odds[2], odds[3]]
        return dfReturn

    # 抓取赔率数据
    i = 2018
    while i > 2004:
        j = 38
        while j >= 1:
            try:
                url = getOddsUrl(i, j)
                print("Get Odds Data from Url[%s]" % (url))
                headers["Referer"] = "http://zq.win007.com/cn/League/%s-%s/36.html" % (i, i + 1)
                print(headers)
                resp = requests.session().get(url, headers=headers, cookies=cookies, timeout=5)
                print(resp.text)
                dfRoundOdds = analysisOddsData(resp.text)
                print(dfRoundOdds.head())
                # 写入文件
                dfRoundOdds.to_csv('./data/ENG_PR/odds.csv', index=False, mode='a', header=False)
                j = j - 1
            except Exception as e:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

            time.sleep(random.randint(3, 5))
        i -= 1
# ===========================赔率数据处理===========================
# 抓取赛程数据
# crwalMatch()
# 抓取赔率数据
# crwalOdds()




