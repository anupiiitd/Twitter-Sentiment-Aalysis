import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob, Word, Blobber
import pandas as pd
import numpy as np
import os
from math import pi
from bokeh.transform import cumsum
import re #regular expressions
from flask import Flask, render_template,request, send_file, make_response, url_for, Response
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from bokeh.models import ColumnDataSource, map_plots
from bokeh.palettes import Spectral6
from bokeh.embed import components
from bokeh.plotting import figure, show
from bokeh.models import FactorRange,HoverTool
from bokeh.io import output_notebook, output_file,show
import random
import urllib.request

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def home():
    if request.method == 'GET':
        return render_template('homepage.html')

@app.route('/feature1',methods = ['GET','POST'])
def index():
    if request.method == 'GET':

        return render_template('primary.html')


    elif request.method =='POST':
        # print("post request made")
        name = request.form['p_name']
        typ = request.form['p_type']
        rtweet = request.form['frt']
        # print(name)
        # print(typ)
        # print(rtweet)
        #Anup
        A_Key="V2MKVagJ7FHniRjtZ6hv1nOx3"
        A_Key_Secret="Wza1nJO6oSAhUryZ68arQDMvEPh3GbhcpBn9kqPHl30uw9gRw9"
        A_Token = "1435660546930475012-fzLwcYBUzeqU4Q29T12mc9JHkn4XgS"
        A_Token_Secret = "c7Sl327JDfIZkHw8s3LENFgH77tKcutM9lcG3m59C9Cxh"
        #Archit
        # A_Key="R1hHLkEWr84GNWySQBE2lWE7N"
        # A_Key_Secret="ersg9u2PCwCbrl59I1Ua581i130892eJFjpNUm37kjvzVi2PTT"
        # A_Token = "922444225-lwQwTdECIQK3JtHHpOjL66FKVDSUeCoxfi9825lH"
        # A_Token_Secret = "8bJRiEul86NwPlvRyk1NiQlkPIN3Rz9YyplVqqRyK3b0H"

        auth = tweepy.OAuthHandler(A_Key,A_Key_Secret)
        auth.set_access_token(A_Token,A_Token_Secret)
        api = tweepy.API(auth)

        def get_tweetsdf(topic):
            tweetList = []
            userList = []
            likesList = []
            datetimeList = []
            locationList = []
            if rtweet=="No":
                Query = topic
            elif rtweet=="Yes":
                Query = topic +  "-filter:retweets"
            cursor = tweepy.Cursor(api.search_tweets,q = Query,lang = "en",tweet_mode = "extended",exclude="retweets").items(150)
            for t in cursor:
                tweetList.append(t.full_text)
                userList.append(t.user.name)
                likesList.append(t.favorite_count)
                locationList.append(t.user.location)
                datetimeList.append(t.created_at)
        
            df =  pd.DataFrame({"User Name":userList,"Tweets":tweetList,"Likes":likesList,
                            "Date Time":datetimeList,"Location":locationList })
            return df

        #sentiment_analysis using TextBlob
        

        def get_polarity_dataframe(df):
        #Section 1
            tempdf = df
            x = tempdf["Tweets"]

            nltk.download('stopwords') #to remove stop words(a,an the)
        
            stop_words=stopwords.words('english') # to get stem of verb(playing->play)
            stemmer=PorterStemmer()

            cleaned_text=[]
            for i in range(len(x)):
                tweet=re.sub('[^a-zA-Z]',' ',x.iloc[i]) #substitute empty string where char is not a-zA-Z
                tweet=tweet.lower().split()

                tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)] #stemming and remove stop words
                tweet=' '.join(tweet)
                cleaned_text.append(tweet)

            tempdf["Clean Tweets"] = cleaned_text
            def sentiment_analysis(df):
                def getSubjectivity(text):
                    return TextBlob(text).sentiment.subjectivity #subjectivity  =1 (personal opinion) =0(factual point)
        
        #Create a function to get the polarity
                def getPolarity(text):
                    return TextBlob(text).sentiment.polarity #polarity lies [-1,1]
        
        #Create two new columns ‘Subjectivity’ & ‘Polarity’
                df["Subjectivity"] =    df["Clean Tweets"].apply(getSubjectivity)
                df["Polarity"] = df["Clean Tweets"].apply(getPolarity)
                def getAnalysis(score):
                    if score <= -0.1:
                        return "negative"
                    elif score > -0.1 and score < 0.1:
                        return "neutral"
                    else:
                        return "positive"
                df["Polarity_Analysis"] = df["Polarity"].apply(getAnalysis )
                return df

        #Section2
            newdf = sentiment_analysis(tempdf)
            return newdf


        pold = get_polarity_dataframe(get_tweetsdf(name))
        #Wordcloud for Mixed Tweets
        all_tweets = []
        for i in range(len(pold)):
            all_tweets.append(pold.loc[i].Tweets)
        text_all = ' '.join(all_tweets)

        proc1 = re.compile('<.*?>')
        def proc(text_in):
            text_o = re.sub(proc1, '', text_in) # Removes HTML Tags and other symbols under proc1
            output = re.sub(r'[^\w\s]', '', text_o) #Removes URLs from the text 
            return output.lower()  #Returns the processed text in lower case

        word_cloud_p_processed = WordCloud(collocations = False, background_color = 'white').generate(proc(text_all)) #Stopwords are removed in this line by 3rd argument
        plt.imshow(word_cloud_p_processed, interpolation='bilinear')
        plt.axis("off")
        
        # plt.show()
        # plt.plot(x,y)

        # plt.savefig(img, format='png')
        # plt.close()
        # img.seek(0)
        # plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # return render_template('plot.html', plot_url=plot_url)

        #Seperation of +ve and negative tweets
        pos = []
        neg = []
        for i in range(len(pold)):
            if pold.loc[i].Polarity_Analysis == "positive":
                pos.append(pold.loc[i].Tweets)
            elif pold.loc[i].Polarity_Analysis == "negative":
                neg.append(pold.loc[i].Tweets)
                
        #Wordcloud for +ve Tweets
        path_static  = os.path.join(os.getcwd(),"static/")
        text_pp = ' '.join(pos)
        word_cloud_p_processed = WordCloud(collocations = False, background_color = 'white').generate(proc(text_pp)) #Stopwords are removed in this line by 3rd argument
        plt.imshow(word_cloud_p_processed, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(path_static+"wordcld_p.png")

        #Wordcloud for Negative Tweets
        text_nn = ' '.join(neg)
        word_cloud_p_processed = WordCloud(collocations = False, background_color = 'white').generate(proc(text_nn)) #Stopwords are removed in this line by 3rd argument
        plt.imshow(word_cloud_p_processed, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(path_static+"wordcld_n.png")

        #To remove \n from the tweets column
        def processTweet(row):
            row["Tweets"] = row["Tweets"].replace("\n","")
            return row
        
        #Top 10 +ve tweet dataframe
        top_pos_10 = []
        for i in range(len(pold)):
            if pold.loc[i].Polarity_Analysis == "positive":
                top_pos_10.append(pold.loc[i])
        t10_p = pd.DataFrame(top_pos_10)
        rslt_df_p = t10_p.sort_values(by = 'Polarity')
        final_p = rslt_df_p.head(10).drop(["Location","Clean Tweets", "Subjectivity", "Polarity", "Polarity_Analysis"], axis=1)
        final_p =  final_p.apply(processTweet, axis =1)


        #Top 10 -ve tweet dataframe
        top_neg_10 = []
        for i in range(len(pold)):
            if pold.loc[i].Polarity_Analysis == "negative":
                top_neg_10.append(pold.loc[i])
        t10_n = pd.DataFrame(top_neg_10)
        rslt_df_n = t10_n.sort_values(by = 'Polarity')
        final_n = rslt_df_n.head(10).drop(["Location","Clean Tweets", "Subjectivity", "Polarity", "Polarity_Analysis"], axis=1)
        final_n =  final_n.apply(processTweet, axis =1)

        def get_polarity_plot(topic):
        #get dataframe from topic
            df1 = get_tweetsdf(topic=topic)

        #get dataframe with polarity columns
            newdf = get_polarity_dataframe(df1)

            posNumber = newdf[newdf["Polarity_Analysis"]=="positive"].shape[0]
            negNumber = newdf[newdf["Polarity_Analysis"]=="negative"].shape[0]
            neuNumber = newdf[newdf["Polarity_Analysis"]=="neutral"].shape[0]

            sentiments = ["Positive","Negative","Neutral"]
            counts = [posNumber, negNumber, neuNumber]
            total = posNumber + negNumber + neuNumber
            if typ == "Pie Chart":
                x = {
                        'Positive': posNumber,
                        'Negative': negNumber,
                        'Neutral': neuNumber,
                    }
                data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
                data['angle'] = data['value']/data['value'].sum() * 2*pi
                data['color'] = ["seagreen","#ffa600","#58508d"]
                # print(data)

                output_file("plot.html")
                p = figure(height=350, title="Pie Chart", toolbar_location=None,
                    tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))
                p.wedge(x=0, y=1, radius=0.4,
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='color', legend_field='country', source=data) 
                p.axis.axis_label = None
                p.axis.visible = False
                p.grid.grid_line_color = None
                p.title.align = "center"
                return p
            elif typ == "Bar Graph":
                source = ColumnDataSource(data=dict(sentiments = sentiments, counts=counts, color=Spectral6))

                p = figure( x_range = sentiments, title=f"Sentiments Plot for latest {total} tweets",
                toolbar_location=None, tools="",x_axis_label = "Sentiments", y_axis_label = "Tweets Count") 
                p.vbar(x='sentiments', top='counts', width=0.8, color='color',  source=source)
                p.xgrid.grid_line_color = None
                p.title.align = "center"
                return p
               
        p1 = get_polarity_plot(str(name))
        demo_script_code,chart_code = components(p1)
        
 
        return render_template('show.html',chart_code = chart_code,demo_script_code = demo_script_code,
        table1=[final_p.to_html(classes='data1', index = False)],
        table2=[final_n.to_html(classes='data2', index = False)],titles= final_p.columns.values)


@app.route('/feature2',methods = ['GET','POST'])
def index2():
    if request.method == 'GET':

        return render_template('primary2.html')

    elif request.method =='POST':
        print("post request made")
        username1 = request.form['p_name1']
        username2 = request.form['p_name2']
        city  = request.form['city']
        country = "India"
        print(username1)
        path_static  = os.path.join(os.getcwd(),"static/")
        #Anup
        A_Key="V2MKVagJ7FHniRjtZ6hv1nOx3"
        A_Key_Secret="Wza1nJO6oSAhUryZ68arQDMvEPh3GbhcpBn9kqPHl30uw9gRw9"
        A_Token = "1435660546930475012-fzLwcYBUzeqU4Q29T12mc9JHkn4XgS"
        A_Token_Secret = "c7Sl327JDfIZkHw8s3LENFgH77tKcutM9lcG3m59C9Cxh"
        #Archit
        # A_Key="R1hHLkEWr84GNWySQBE2lWE7N"
        # A_Key_Secret="ersg9u2PCwCbrl59I1Ua581i130892eJFjpNUm37kjvzVi2PTT"
        # A_Token = "922444225-lwQwTdECIQK3JtHHpOjL66FKVDSUeCoxfi9825lH"
        # A_Token_Secret = "8bJRiEul86NwPlvRyk1NiQlkPIN3Rz9YyplVqqRyK3b0H"

        auth = tweepy.OAuthHandler(A_Key,A_Key_Secret)
        auth.set_access_token(A_Token,A_Token_Secret)
        api = tweepy.API(auth)

        def get_geo_id(city, country):
            geo = api.search_geo(query = city, granularity = "city", max_results =4)
            for i in range(len(geo)):
                id = geo[i].id
                if country.lower() == geo[i].country.lower():
                    return id

       

        def get_tweetsdf(username, city, country):
            tweetList = []
            userList = []
            likesList = []
            datetimeList = []
            locationList = []
            geo_id = get_geo_id(city, country)
            Query = "(-filter:retweets @{screenname} place:{geo} lang:en)".format(geo = geo_id,screenname = username)
            cursor = tweepy.Cursor(api.search_tweets,q = Query,lang = "en",tweet_mode = "extended").items(150)
            for t in cursor:
                tweetList.append(t.full_text)
                userList.append(t.user.name)
                likesList.append(t.favorite_count)
                locationList.append(t.user.location)
                datetimeList.append(t.created_at)
        
            df =  pd.DataFrame({"User Name":userList,"Tweets":tweetList,"Likes":likesList,
                            "Date Time":datetimeList,"Location":locationList })
            return df

        #sentiment_analysis using TextBlob
        

        def get_polarity_dataframe(df):
        #Section 1
            tempdf = df
            x = tempdf["Tweets"]

            nltk.download('stopwords') #to remove stop words(a,an the)
        
            stop_words=stopwords.words('english') # to get stem of verb(playing->play)
            stemmer=PorterStemmer()

            cleaned_text=[]
            for i in range(len(x)):
                tweet=re.sub('[^a-zA-Z]',' ',x.iloc[i]) #substitute empty string where char is not a-zA-Z
                tweet=tweet.lower().split()

                tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)] #stemming and remove stop words
                tweet=' '.join(tweet)
                cleaned_text.append(tweet)

            tempdf["Clean Tweets"] = cleaned_text
            def sentiment_analysis(df):
                def getSubjectivity(text):
                    return TextBlob(text).sentiment.subjectivity #subjectivity  =1 (personal opinion) =0(factual point)
        
        #Create a function to get the polarity
                def getPolarity(text):
                    return TextBlob(text).sentiment.polarity #polarity lies [-1,1]
        
        #Create two new columns ‘Subjectivity’ & ‘Polarity’
                df["Subjectivity"] =    df["Clean Tweets"].apply(getSubjectivity)
                df["Polarity"] = df["Clean Tweets"].apply(getPolarity)
                def getAnalysis(score):
                    if score <= -0.1:
                        return "negative"
                    elif score > -0.1 and score < 0.1:
                        return "neutral"
                    else:
                        return "positive"
                df["Polarity_Analysis"] = df["Polarity"].apply(getAnalysis )
                return df

        #Section2
            newdf = sentiment_analysis(tempdf)
            return newdf


        df1_polarity = get_polarity_dataframe(get_tweetsdf(username1,city=city,country=country))
        df2_polarity = get_polarity_dataframe(get_tweetsdf(username2,city=city,country=country))
        

        #To remove \n from the tweets column
        def processTweet(row):
            row["Tweets"] = row["Tweets"].replace("\n","")
            return row    

        def get_positive_negative_dataframes(pold): #pold is dataframe with polarity_analysis column
            #Top 10 +ve tweet dataframe
            top_pos_10 = []
            for i in range(len(pold)):
                if pold.loc[i].Polarity_Analysis == "positive":
                    top_pos_10.append(pold.loc[i])
            t10_p = pd.DataFrame(top_pos_10)
            rslt_df_p = t10_p.sort_values(by = 'Polarity')
            final_p = rslt_df_p.head(10).drop(["Location","Clean Tweets", "Subjectivity", "Polarity", "Polarity_Analysis"], axis=1)
            final_p =  final_p.apply(processTweet, axis =1)

            #Top 10 -ve tweet dataframe
            top_neg_10 = []
            for i in range(len(pold)):
                if pold.loc[i].Polarity_Analysis == "negative":
                    top_neg_10.append(pold.loc[i])
            t10_n = pd.DataFrame(top_neg_10)
            rslt_df_n = t10_n.sort_values(by = 'Polarity')
            final_n = rslt_df_n.head(10).drop(["Location","Clean Tweets", "Subjectivity", "Polarity", "Polarity_Analysis"], axis=1)
            final_n =  final_n.apply(processTweet, axis =1)

            return final_p, final_n

        def save_profile_pic(username,filename,api):
            t2 = api.search_users(q  = "@{user}".format(user = username))
            if (t2[0]._json["default_profile_image"]==False):
                pic_url = t2[0]._json["profile_image_url"] 
                pic_url = pic_url.replace("_normal","") #to get good resolution pic
            if (t2[0]._json["default_profile_image"]==True):
                pic_url = "https://www.pngitem.com/pimgs/m/150-1503945_transparent-user-png-default-user-image-png-png.png"                
            urllib.request.urlretrieve(pic_url, path_static+"/"+filename)
            pass
        
        def get_name(username):
            t2 = api.search_users(q  = "@{user}".format(user = username))
            return t2[0]._json["name"]

        def get_polarity_plot_compare(df1,df2):
            def get_sentiment_counts(newdf):
                posNumber = newdf[newdf["Polarity_Analysis"]=="positive"].shape[0]
                neuNumber = newdf[newdf["Polarity_Analysis"]=="neutral"].shape[0]
                negNumber = newdf[newdf["Polarity_Analysis"]=="negative"].shape[0]
                
                counts = [posNumber, neuNumber, negNumber]
                return counts

            user1_sentiment_counts = get_sentiment_counts(df1)
            user2_sentiment_counts = get_sentiment_counts(df2)
            total = sum(user1_sentiment_counts)

            c1 = user1_sentiment_counts
            c2 = user2_sentiment_counts

            name1 = get_name(username1)
            name2 = get_name(username2)

            #List of persons
            names = [name1,name2]
            sentiments = ["Positive","Neutral","Negative"]

            #Creating color palette of 6 colors
            user_palette = ('#238723','#5bd446','#225fba',"#43b0f0",'#821616', "#d62728")

            #Creating a dictionary of our data
            mdata = {'sentiments' : sentiments,
                    name1 : c1,
                    name2 : c2
                    }
            # Creating tuples for individual bars 
            x = [ (sentiment, person) for sentiment in sentiments for person in names ]
            counts = sum(zip(mdata[name1], mdata[name2]), ())
            #Creating a column data source - Bokeh's own data type 
            source = ColumnDataSource(data=dict(x=x, counts=counts, color=user_palette))
            #Initializing our plot
            p = figure(x_range=FactorRange(*x), plot_height=500, title="Sentiments comparison",plot_width=800)
            #Plotting our vertical bar chart
            p.vbar(x='x', top='counts', width=0.9  ,fill_color='color',  source=source)
            #Enhancing our graph
            p.y_range.start = 0
            p.x_range.range_padding = 0.1
            p.xaxis.major_label_orientation = .9
            p.xgrid.grid_line_color = None
            p.title.align = "center"
            p.title.text_font_size = "20px"
            p.yaxis.axis_label = "Tweet Count"

            p.add_tools(HoverTool(tooltips=[("For","@x"), ("Count", "@counts")]))
            return p



        #To get polarity tables
        user1_pos_df, user1_neg_df = get_positive_negative_dataframes(df1_polarity)
        user2_pos_df, user2_neg_df = get_positive_negative_dataframes(df2_polarity)
        # print("Person 1 -ve df")
        # print(user1_neg_df)
        # print("Person 1 +ve df")
        # print(user1_pos_df)
        #To save_profile_pic
        save_profile_pic(username1, "feature2_user1.jpg",api)
        save_profile_pic(username2, "feature2_user2.jpg",api)

        #To get names
        name1 = get_name(username1)
        name2 = get_name(username2)

               
        p1 = get_polarity_plot_compare(df1_polarity, df2_polarity)
        demo_script_code,chart_code = components(p1)
 
        return render_template('show2.html',chart_code = chart_code,demo_script_code = demo_script_code,personName1 = name1, personName2 = name2,
        table1_1=[user1_pos_df.to_html(classes='data', index = False)],table2_1=[user2_pos_df.to_html(classes='data', index = False)],
        table1_2=[user1_neg_df.to_html(classes='data', index = False)],
        table2_2=[user2_neg_df.to_html(classes='data', index = False)],
        titles= user1_pos_df.columns.values,
        city = city)


if __name__ == "__main__":
    app.run(debug=True)
