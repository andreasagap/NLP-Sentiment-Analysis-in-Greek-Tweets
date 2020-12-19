import snscrape.modules.twitter as sntwitter
import csv
from datetime import date, timedelta
import twint
import time

def getTweets():
    csvFile = open('dataset.csv', 'a',encoding='utf-8', newline='')
    csvWriter = csv.writer(csvFile,delimiter ='~')
    maxTweets = 100  # the number of tweets you require
    start_date = date(2020, 11, 10)
    end_date = date(2020, 12, 1)
    delta = timedelta(days=5)
    while start_date <= end_date:
        since = start_date.strftime("%Y-%m-%d")
        start_date += delta
        until = start_date.strftime("%Y-%m-%d")
        for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper('#Covid19 #Covid_19' + ' since:' + since +  " until:" +until +" lang:el").get_items()):
            if i > maxTweets:
                break
            csvWriter.writerow([str(tweet.date.day) + "/" + str(tweet.date.month) + "/" + str(tweet.date.year), tweet.content.replace("\n","")])
        print(since)

def getTweets2():
    c = twint.Config()
    c.Search = "#Covid19 OR #covid19 OR #lockdown AND κυβερνηση"
    c.Lang = "el"
    c.Limit = 1000
    c.Popular_tweets = True
    #c.Location = True
    c.Store_csv = True
    c.Output = "dataset_v2.csv"
    start_date = date(2020, 4, 1)
    end_date = date(2020, 12, 1)
    delta = timedelta(days=2)
    while start_date <= end_date:
        since = start_date.strftime("%Y-%m-%d")
        start_date += delta
        until = start_date.strftime("%Y-%m-%d")
        c.Until = until
        c.Since = since
        twint.run.Search(c)
        time.sleep(10)

if __name__ == '__main__':
    getTweets2()

