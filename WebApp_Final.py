import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
def app():

  
   st.write("""
   # Video Game Sales Prediction
   Predict the sales of your video games!
   """)

   df = pd.read_csv('/data/videogames.csv')

   st.subheader('Data Information:')
   st.dataframe(df)
   st.write(df.describe())

   ###### INSERT GRAPH EXPLORATION
   x_opts = ['Platform', 'Year', 'Genre', 'Publisher']
   x_axis = st.selectbox('Which category do you want to explore?', x_opts)
   st.subheader('Statistical Exploration:')
   fig = px.scatter(df, x=x_axis, y='Global_Sales', hover_name=f'{x_axis}')
   st.plotly_chart(fig)

   df = df.dropna(subset=['Publisher'])

   df1 = df[['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales']]
   number = LabelEncoder()
   df['Platform'] = number.fit_transform(df['Platform'].astype('str'))
   df['Genre'] = number.fit_transform(df['Genre'].astype('str'))
   df['Publisher'] = number.fit_transform(df['Publisher'].astype('str'))

   x = df[['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales']]
   y = df['Global_Sales']

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

   scaler = StandardScaler()

   scaler.fit(x_train)
   x_train = scaler.transform(x_train)
   x_test = scaler.transform(x_test)

   regr = linear_model.LinearRegression()
   regr.fit(x_train, y_train)

   def get_user_input():
      plat = ['2600', '3DO', '3DS', 'DC', 'DS', 'GB', 'GBA', 'GC', 'GEN', 'GG',
              'N64', 'NES', 'NG', 'PC', 'PCFX', 'PS', 'PS2', 'PS3', 'PS4', 'PSP',
              'PSV', 'SAT', 'SCD', 'SNES', 'TG16', 'WS', 'Wii', 'WiiU', 'X360',
              'XB', 'XOne']
      platform = st.sidebar.selectbox('Platform', plat)
      gen = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle',
             'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports',
             'Strategy']
      genre = st.sidebar.selectbox('Genre', gen)
      pub = ['10TACLE Studios', '1C Company', '20th Century Fox Video Games',
             '2D Boy', '3DO', '49Games', '505 Games', '5pb', '7G//AMES',
             '989 Sports', '989 Studios', 'AQ Interactive', 'ASC Games',
             'ASCII Entertainment', 'ASCII Media Works', 'ASK', 'Abylight',
             'Acclaim Entertainment', 'Accolade', 'Ackkstudios', 'Acquire',
             'Activision', 'Activision Blizzard', 'Activision Value',
             'Adeline Software', 'Aerosoft', 'Agatsuma Entertainment', 'Agetec',
             'Aksys Games', 'Alawar Entertainment', 'Alchemist',
             'Alternative Software', 'Altron', 'Alvion', 'American Softworks',
             'Angel Studios', 'Answer Software', 'Aqua Plus', 'Aques',
             'Arc System Works', 'Arena Entertainment', 'Aria', 'Arika',
             'ArtDink', 'Aruze Corp', 'Ascaron Entertainment',
             'Ascaron Entertainment GmbH', 'Asgard', 'Asmik Ace Entertainment',
             'Asmik Corp', 'Aspyr', 'Astragon', 'Asylum Entertainment', 'Atari',
             'Athena', 'Atlus', 'Avalon Interactive', 'Avanquest',
             'Avanquest Software', 'Axela', 'BAM! Entertainment',
             'BMG Interactive Entertainment', 'BPS', 'Banpresto', 'Benesse',
             'Berkeley', 'Bethesda Softworks', 'Big Ben Interactive',
             'Big Fish Games', 'Bigben Interactive', 'Black Bean Games',
             'Black Label Games', 'Blast! Entertainment Ltd', 'Blue Byte',
             'Bohemia Interactive', 'Bomb', 'Boost On', 'Brash Entertainment',
             'Broccoli', 'BushiRoad', 'CBS Electronics', 'CCP',
             'CDV Software Entertainment', 'CPG Products', 'CTO SpA', 'Capcom',
             'Cave', 'ChunSoft', 'City Interactive',
             'Cloud Imperium Games Corporation', 'Coconuts Japan',
             'Codemasters', 'Codemasters Online', 'CokeM Interactive', 'Coleco',
             'Comfort', 'Commseed', 'Compile', 'Compile Heart',
             'Conspiracy Entertainment', 'Core Design Ltd.',
             'Crave Entertainment', 'Creative Core', 'Crimson Cow',
             'Crystal Dynamics', 'Culture Brain', 'Culture Publishers',
             'CyberFront', 'Cygames', 'D3Publisher', 'DHM Interactive',
             'DSI Games', 'DTP Entertainment', 'Daedalic',
             'Daedalic Entertainment', 'Daito', 'Data Age',
             'Data Design Interactive', 'Data East', 'Datam Polystar',
             'Deep Silver', 'Destination Software, Inc', 'Destineer',
             'Detn8 Games', 'Devolver Digital', 'DigiCube',
             'Disney Interactive Studios', 'Dorart', 'DreamCatcher Interactive',
             'DreamWorks Interactive', 'Dusenberry Martin Racing', 'EA Games',
             'EON Digital Entertainment', 'ESP', 'Easy Interactive', 'Ecole',
             'Edia', 'Eidos Interactive', 'Electronic Arts',
             'Electronic Arts Victor', 'Elf', 'Elite', 'Empire Interactive',
             'Encore', 'Enix Corporation', 'Enjoy Gaming ltd.', 'Enterbrain',
             'Epic Games', 'Epoch', 'Ertain', 'Essential Games',
             'Evolution Games', 'Evolved Games', 'Excalibur Publishing',
             'Experience Inc.', 'Extreme Entertainment Group',
             'Falcom Corporation', 'Fields', 'Flashpoint Games', 'Flight-Plan',
             'Focus Home Interactive', 'Focus Multimedia',
             'Foreign Media Games', 'Fortyfive', 'Fox Interactive',
             'From Software', 'FuRyu', 'FuRyu Corporation', 'Fuji', 'FunSoft',
             'Funbox Media', 'Funcom', 'Funsta', 'G.Rev', 'GN Software', 'GOA',
             'GSP', 'GT Interactive', 'Gaga', 'Gainax Network Systems',
             'Gakken', 'Game Arts', 'Game Factory', 'Game Life',
             'GameMill Entertainment', 'GameTek', 'Gamebridge', 'Gamecock',
             'Gameloft', 'Gathering of Developers', 'General Entertainment',
             'Genki', 'Genterprise', 'Ghostlight', 'Giga', 'Giza10', 'Glams',
             'Global A Entertainment', 'Global Star', 'Gotham Games',
             'Graffiti', 'Grand Prix Games', 'Graphsim Entertainment',
             'Gremlin Interactive Ltd', 'Griffin International', 'Groove Games',
             'GungHo', 'Gust', 'HAL Laboratory', 'HMH Interactive', 'Hackberry',
             'Hamster Corporation', 'Happinet', 'Harmonix Music Systems',
             'Hasbro Interactive', 'Havas Interactive', 'Headup Games',
             'Hearty Robin', 'Hect', 'Hello Games', 'Her Interactive',
             'Hip Interactive', 'Home Entertainment Suppliers',
             'Hudson Entertainment', 'Hudson Soft', 'Human Entertainment',
             'HuneX', 'IE Institute', 'ITT Family Games', 'Iceberg Interactive',
             'Idea Factory', 'Idea Factory International',
             'Ignition Entertainment', 'Illusion Softworks', 'Imadio',
             'Image Epoch', 'Imageworks', 'Imagic', 'Imagineer', 'Imax',
             'Indie Games', 'Infogrames', 'Insomniac Games', 'Interchannel',
             'Interchannel-Holon', 'Intergrow', 'Interplay',
             'Interplay Productions', 'Interworks Unlimited, Inc.',
             'Inti Creates', 'Introversion Software',
             'Irem Software Engineering', 'Ivolgamus', 'JVC',
             'Jack of All Games', 'Jaleco', 'Jester Interactive',
             'JoWood Productions', 'Jorudan', 'Just Flight', 'KID', 'KSS',
             'Kadokawa Games', 'Kadokawa Shoten', 'Kaga Create',
             'Kalypso Media', 'Kamui', 'Kando Games', 'Karin Entertainment',
             'Kemco', 'Kids Station', 'King Records', 'Knowledge Adventure',
             'Koch Media', 'Kokopeli Digital Studios',
             'Konami Digital Entertainment', 'Kool Kizz', 'LEGO Media',
             'LSP Games', 'Laguna', 'Legacy Interactive', 'Level 5',
             'Lexicon Entertainment', 'Licensed 4U', 'Lighthouse Interactive',
             'Liquid Games', 'Little Orbit', 'Locus', 'LucasArts',
             'MC2 Entertainment', 'MLB.com', 'MTO', 'MTV Games', 'Mad Catz',
             'Magical Company', 'Magix', 'Majesco Entertainment', 'Mamba Games',
             'Marvel Entertainment', 'Marvelous Entertainment',
             'Marvelous Games', 'Marvelous Interactive', 'Masque Publishing',
             'Mastertronic', 'Mastiff', 'Mattel Interactive', 'Max Five',
             'Maximum Family Games', 'Maxis', 'Media Entertainment',
             'Media Factory', 'Media Rings', 'Media Works', 'MediaQuest',
             'Men-A-Vision', 'Mentor Interactive', 'Mercury Games',
             'Merscom LLC', 'Metro 3D', 'Michaelsoft', 'Micro Cabin',
             'Microids', 'Microprose', 'Microsoft Game Studios',
             'Midas Interactive Entertainment', 'Midway Games', 'Milestone',
             'Milestone S.r.l', 'Milestone S.r.l.', 'Minato Station',
             'Mindscape', 'Mirai Shounen', 'Misawa', 'Mitsui', 'Mojang',
             'Monte Christo Multimedia', 'Moss', 'Mud Duck Productions',
             'Mumbo Jumbo', 'Mycom', 'Myelin Media', 'Mystique', 'NCS',
             'NCSoft', 'NDA Productions', 'NEC', 'NEC Interchannel',
             'Namco Bandai Games', 'Natsume', 'Navarre Corp', 'Naxat Soft',
             'Neko Entertainment', 'NetRevo', 'New', 'New World Computing',
             'NewKidCo', 'Nexon', 'Nichibutsu', 'Nihon Falcom Corporation',
             'Nintendo', 'Nippon Amuse', 'Nippon Columbia',
             'Nippon Ichi Software', 'Nippon Telenet', 'Nitroplus', 'Nobilis',
             'Nordcurrent', 'Nordic Games', 'NovaLogic', 'Number None',
             'O-Games', 'O3 Entertainment', 'Ocean', 'Office Create',
             'On Demand', 'Ongakukan', 'Origin Systems', 'Otomate',
             'Oxygen Interactive', 'P2 Games', 'PM Studios', 'PQube',
             'Pacific Century Cyber Works', 'Pack In Soft', 'Pack-In-Video',
             'Palcom', 'Panther Software', 'Paon', 'Paon Corporation',
             'Paradox Development', 'Paradox Interactive', 'Parker Bros.',
             'Performance Designed Products', 'Phantagram', 'Phantom EFX',
             'Phenomedia', 'Phoenix Games', 'Piacci', 'Pinnacle', 'Pioneer LDC',
             'Play It', 'PlayV', 'Playlogic Game Factory', 'Playmates',
             'Playmore', 'Plenty', 'Pony Canyon', 'PopCap Games',
             'PopTop Software', 'Popcorn Arcade', 'Pow', 'Princess Soft',
             'Prototype', 'Psygnosis', 'Quelle', 'Quest', 'Quinrose', 'Quintet',
             'RED Entertainment', 'RTL', 'Rage Software', 'Rain Games',
             'Rebellion', 'Rebellion Developments', 'Red Orb',
             'Red Storm Entertainment', 'RedOctane', 'Reef Entertainment',
             'Revolution (Japan)', 'Revolution Software', 'Rising Star Games',
             'Riverhillsoft', 'Rocket Company', 'Rondomedia', 'Russel',
             'SCS Software', 'SCi', 'SNK', 'SNK Playmore', 'SPS', 'SSI',
             'Sammy Corporation', 'Saurus', 'Scholastic Inc.', 'Screenlife',
             'Sears', 'Sega', 'Seta Corporation', 'Seventh Chord', 'Shogakukan',
             'Simon & Schuster Interactive', 'Slightly Mad Studios',
             'Slitherine Software', 'Societa', 'Sold Out', 'Sonnet',
             'Sony Computer Entertainment',
             'Sony Computer Entertainment America',
             'Sony Computer Entertainment Europe', 'Sony Music Entertainment',
             'Sony Online Entertainment', 'SouthPeak Games', 'Spike', 'Square',
             'Square EA', 'Square Enix', 'SquareSoft', 'Stainless Games',
             'Starfish', 'Starpath Corp.', 'Sting', 'Storm City Games',
             'Strategy First', 'Success', 'Summitsoft', 'Sunflowers',
             'Sunrise Interactive', 'Sunsoft', 'Sweets', 'Swing! Entertainment',
             'Syscom', 'System 3', 'System 3 Arcade Software', 'System Soft',
             'T&E Soft', 'TDK Core', 'TDK Mediactive', 'TGL', 'THQ', 'TOHO',
             'TYO', 'Taito', 'Takara', 'Takara Tomy', 'Take-Two Interactive',
             'Takuyo', 'TalonSoft', 'Team17 Software', 'TechnoSoft',
             'Technos Japan Corporation', 'Tecmo Koei', 'Telegames',
             'Telltale Games', 'Telstar', 'Tetris Online',
             'The Adventure Company', 'The Learning Company', 'Tigervision',
             'Time Warner Interactive', 'Titus', 'Tivola', 'Tommo',
             'Tomy Corporation', 'TopWare Interactive', 'Touchstone',
             'Tradewest', 'Trion Worlds', 'Tripwire Interactive',
             'Tru Blu Entertainment', 'Tryfirst', 'Type-Moon', 'U.S. Gold',
             'UEP Systems', 'UFO Interactive', 'UIG Entertainment', 'Ubisoft',
             'Ubisoft Annecy', 'Ultravision', 'Universal Gamex',
             'Universal Interactive', 'Unknown', 'Valcon Games', 'ValuSoft',
             'Valve', 'Valve Software', 'Vap', 'Vatical Entertainment',
             'Vic Tokai', 'Victor Interactive', 'Video System', 'Views',
             'Vir2L Studios', 'Virgin Interactive', 'Virtual Play Games',
             'Visco', 'Vivendi Games', 'Wanadoo', 'Warashi', 'Wargaming.net',
             'Warner Bros. Interactive Entertainment', 'Warp',
             'WayForward Technologies', 'Westwood Studios',
             'White Park Bay Software', 'Wizard Video Games', 'XS Games',
             'Xicat Interactive', 'Xing Entertainment', 'Xplosiv',
             'Xseed Games', 'Yacht Club Games', 'Yamasa Entertainment', 'Yeti',
             "Yuke's", 'Yumedia', 'Zenrin', 'Zoo Digital Publishing',
             'Zoo Games', 'Zushi Games', 'bitComposer Games', 'dramatic create',
             'fonfun', 'iWin', 'id Software', 'imageepoch Inc.',
             'inXile Entertainment', 'mixi, Inc', 'responDESIGN']
      publisher = st.sidebar.selectbox('Publisher', pub)

      user_data = {'Platform': platform, 'Genre': genre, 'Publisher': publisher}
      features = pd.DataFrame(user_data, index=[0])
      return features

   user_input = get_user_input()

   st.subheader('User Values:')
   ##st.write(type(user_input['Platform'][0]))
   st.write(user_input)
   grouped_multiple = df1.groupby(['Platform', 'Genre']).agg({'NA_Sales': ['mean'], 'EU_Sales': ['mean']})
   grouped_multiple.columns = ['NA_mean', 'EU_mean']
   grouped_multiple = grouped_multiple.reset_index()
   ##st.write(grouped_multiple.head())

   dum = grouped_multiple.loc[((grouped_multiple['Platform'] == user_input['Platform'][0]) & (
              grouped_multiple['Genre'] == user_input['Genre'][0]))]
   in1 = dum.index.values[0]
   ##st.write(dum)
   user_input['NA_Sales'] = dum['NA_mean'][in1]
   user_input['EU_Sales'] = dum['EU_mean'][in1]

   ##st.subheader('User Values###:')
   ##st.write(type(user_input['Platform'][0]))
   ##st.write(user_input)

   ind = df1.loc[(df1['Platform']) == user_input['Platform'][0]].index.values[0]
   ind2 = df1.loc[(df1['Genre']) == user_input['Genre'][0]].index.values[0]
   ind3 = df1.loc[(df1['Publisher']) == user_input['Publisher'][0]].index.values[0]
   ##st.write(df.loc[ind])
   user_input['Platform'] = df.loc[ind]['Platform']
   user_input['Genre'] = df.loc[ind2]['Genre']
   user_input['Publisher'] = df.loc[ind3]['Publisher']
   ##st.subheader('User Values:')
   ##st.write(type(user_input['Platform'][0]))
   ##st.write(user_input)
   pred = regr.predict(user_input)

   st.subheader('Predicted Sales in Millions:')
   st.write(pred[0])