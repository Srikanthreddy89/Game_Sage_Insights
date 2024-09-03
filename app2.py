import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import io
from PIL import Image
def app():
    st.title('VIDEO GAME RECOMMENDATION SYSTEM')
    #st.write('Welcome to app2')
    df = pd.read_csv('/data/steam_data.csv')
    df = df[df['positive_ratings'] > 1000]
    df = df.reset_index(drop=True)
    columns = ['steamspy_tags', 'name']
    df[columns].head(30)

    def get_important_features(data):
        important_features = []
        for i in range(0, data.shape[0]):
            important_features.append(data['steamspy_tags'][i] + ' ' + data['name'][i])

        return important_features

    df['important_features'] = get_important_features(df)
    df['Game_no'] = range(0, df.shape[0])

    cm = CountVectorizer().fit_transform(df['important_features'])
    cs = cosine_similarity(cm)

    cho = ['Genre Wise', 'Title Wise']
    selected_choice = st.sidebar.selectbox('Choose the mode of recommendation', cho)




    def get_user_input():
        title = ['Counter-Strike', 'Grand Theft Auto V', 'Need for Speed: Shift', 'Getting Over It with Bennett Foddy', 'WWE 2K16', 'Assassins CreedÂ® Origins', 'Rust', 'RESIDENT EVIL 7 biohazard / BIOHAZARD 7 resident evil', 'The Room', 'NARUTO SHIPPUDEN: Ultimate Ninja STORM 3 Full Burst HD',
                'Sherlock Holmes: Crimes and Punishments', 'Human: Fall Flat', 'Hitman 2: Silent Assassin', 'Spec Ops: The Line', 'Mortal Kombat X']
        selected_title = st.sidebar.selectbox('Choose Title', title)

        return selected_title

    def get_genre():
        gen = ['Action', 'Battle Royale', 'Comedy', 'Detective', 'Fighting', 'Funny', 'Horror', 'Multiplayer',
               'Racing', 'Strategy', 'Open World', 'FPS', 'Hacking', 'Sports', 'Sniper']
        genre = st.sidebar.selectbox('Choose Genre', gen)

        return genre

    def choicename(choice, tag, name):
        if choice in tag:
            return name

    if(selected_choice=='Title Wise'):
        userchoice = get_user_input()
        app_id = df[df.name == userchoice]['Game_no'].values[0]
        scores = list(enumerate(cs[app_id]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_scores = sorted_scores[1:]
        df['positive_ratings'] = df['positive_ratings'].astype(str)
        j = 0
        st.write('**Top 10 recommended games for **',userchoice,'** are:**')
        st.text("")
        for item in sorted_scores:
            game_title = (df[df.Game_no == item[0]]['name'].values[0] + ' (Postitive Ratings : ' +
                          df[df.Game_no == item[0]]['positive_ratings'].values[0] + ')')
            st.write(game_title)
            st.text("")
            # print('Similarity : ', end='')
            # print(item[1])
            j = j + 1
            if j > 9:
                break

    else:
        userchoice = get_genre()
        st.write('**Top 10 recommended games for **',userchoice,'** are:**')  
        dff = df[df.apply(lambda x: choicename(userchoice, x['steamspy_tags'], x['name']), axis=1).notnull()].sort_values('positive_ratings', ascending=False)['name'].reset_index()
        i=0
        tot = 10
        if(len(dff.index)<10):
            tot = len(dff.index)
        for i in range(tot):
            st.write(dff['name'].loc[i])
            st.text("")






    









