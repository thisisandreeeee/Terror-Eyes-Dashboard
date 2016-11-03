# -*- coding: utf-8 -*-
from keys import * #accesstoken
import tweepy,wgetter
import csv,os,sys
import classify_image as ci

class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        #Check if CSV exists. Else, create it.
        if os.path.isfile('csv-files/terrortracking.csv') == False:
            with open('csv-files/terrortracking.csv','w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(['Screen name','Created At','Status','Location','Lat','Long','Media link'])
                f.close()

        with open('csv-files/terrortracking.csv', 'a',newline="") as f:
            if '@terrorbgone'in status.text.lower():#hack to filter
                writer = csv.writer(f)
                try:
                    lat=status.coordinates['coordinates'][1] #FIXME: 1, 0
                    long=status.coordinates['coordinates'][0]
                except:
                    lat=''
                    long=''
                try:
                    geo = status.place.name
                except:
                    geo=''
                media = status.entities.get('media', [])
                if(len(media) > 0):
                    media=media[0]['media_url']
                    #name=str(status.created_at)+'_'+status.author.screen_name
                    #name += self.extensionFinder(media)
                    wgetter.download(media,outdir="TerrorAttachment")
                writer.writerow([status.author.screen_name, status.created_at, status.text,geo,lat,long,media])
                print("Downloaded! Running classifier..")
                ci.imageClassify("TerrorAttachment")
            f.close()
            
    def on_error(self, status_code):
        print ( sys.stderr, 'Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print ( sys.stderr, 'Timeout...')
        return True # Don't kill the stream

"""
Main twitter function. Creates the customstream listener class and begins streaming tweets.
"""
def twitterCatcherStream():
     print('Beginning Twitter Crowdsource Bot')
     global accesstokenlist
     currentKey = accesstokenlist
     auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
     auth.set_access_token(currentKey[2], currentKey[3])
     api = tweepy.API(auth)
     l=CustomStreamListener()
     stream = tweepy.Stream(api.auth, l)
     stream.userstream('terrorBgone')
     stream.filter(track=['@terrorBgone'],async=True)
