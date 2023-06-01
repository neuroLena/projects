import streamlit as st
import pandas as pd

import numpy as np

# for classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import fastai
from fastai import *
from fastai.text import * 
from fastai.text.all import *

# for sentiment
import matplotlib.pyplot as plt
from textblob import TextBlob


##----------------------------------------------------------------

st.set_page_config(page_title="MUBERT", page_icon="🧿")

data_path='/Users/mymac/Documents/GitHub/streamlit_apps/portfolio/resources/mubert'
with open(f'{data_path}/prepair_clean.md', "r") as file:
    clean_md = file.read()
with open(f'{data_path}/prepair_balance.md', "r") as file:
    balance_md = file.read()
        
clean_df = pd.read_csv(f'{data_path}/clean_df.csv')
balanced_df = pd.read_csv(f'{data_path}/balanced_df.csv')

@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Will only run once if already cached
clean_df = load_data(f'{data_path}/clean_df.csv')
balanced_df = df = load_data(f'{data_path}/balanced_df.csv')

# df = pd.DataFrame(clean_df)


##----------------------------------------------------------------


## TABs ----------------------------------------------------------
tabnames = ["Preprocess Data",
			"EDA", 
			"Predict Downloads",
			"Sentiment Analysis", 
			"Other"
			]
tabs = st.tabs(tabnames)

## Tab: Preprocess Data ------------------------------------------
with tabs[0]:
    st.header('Load original data')
    with st.echo():
        # Load original files
        dir_path=data_path
        data = pd.read_csv(f'{dir_path}/ttm_dump.csv')
        tags = pd.read_json(f'{dir_path}/sid_tags.json', orient='index')
        st.write(f'ttm_dump size: {data.shape}')
        st.dataframe(data.head(3))
        st.write(f'sid_tags size: {tags.shape}')
        st.dataframe(tags.head(3))
    with st.echo():
        # Rename columns
        data = data.rename(columns={'stream_id': 'sid'})
        data = data.rename(columns={'downloaded_count': 'downloaded'})
        data = data.rename(columns={'name': 'input'})
        # Sids to column
        tags['sid']=tags.index
        # Tags columns to list
        tags = pd.DataFrame(tags)
        tags['tags'] = tags.apply(lambda row: [val for val in row.values if val is not None], axis=1)
        tags['tags'] = tags['tags'].astype(str)
        # Keep only ecessary columns
        tags_df = pd.DataFrame(tags, columns=['sid', 'tags'])
        data_df = pd.DataFrame(data) 
    st.header('Merge and clean')
    with st.echo():
        # Top 5 sids
        st.write('Top-5 sids:')
        data_df['sid'].value_counts().iloc[:5]
        # Check if any sid has no tags = no such sid
        n_notags=0
        for i in range(len(tags.index)):
            if not tags.iloc[100].isna().any():
                n_notags+=1
            else:
                pass
        st.write(f'Sids with no tags: {n_notags}')
        # Check missing sids 
        n_nosid=sum(data_df['sid'].isna())
        # tags['sid'].unique() #= no missing sids
        st.write(f'Rows in *data* with no sids: {n_nosid}')
        # Check if there are sids in @tags that are not present in @data 
        set1=set(data_df['sid'][data_df['sid'].notnull()].unique().astype(int))
        set2=set(tags_df['sid'])
        diff12 = set1.difference(set2) #= [76488, 81786, 81923, 81938, 81985, 82038, 82099]
        st.write(f'Sids in *data* not present in *tags*: {list(diff12)}')
        # Check if there are sids in 'tags' hat are not present in 'data'
        diff21=set2.difference(set1) #= all from 'data' are in 'tags'
        st.write(f'Sids in *tags* not present in *data*:{list(diff21)}')
    with st.echo():
        # Exclude nonoverlaping sids
        data_df = data_df[~data_df['sid'].isin(diff12)]
        # Subset data with sids
        nosid_df = data[data['sid'].isna()]
        sid_df = data[data['sid'].notnull()]
        # Merge  datasets by 'sid' column
        merged_df = pd.merge(sid_df, tags_df, on='sid', how='left')
        # Cleaned version
        clean_df = merged_df
        clean_df = clean_df.rename(columns={'sid_x': 'sid'})
        # Check how many rows have >1 downloads
        more1=sum(merged_df['downloaded']>1)
        st.write(f'Rows with more then 1 downloads: {more1}')
        # Change values >1 to 1 in 'downloaded' column
        merged_df.loc[merged_df['downloaded'] > 1, 'downloaded'] = 1  
        clean_df = clean_df[merged_df['downloaded'].isin([0, 1])]
        clean_df.iloc[:,[7,4]]= clean_df.iloc[:,[7,4]].astype(int)
        st.write(f'*clean_df* size: {clean_df.shape}')
        clean_df=pd.DataFrame(clean_df)
        clean_df
        # st.dataframe(clean_df.head())
        # Save/Load data
        # nosid_df.to_csv(f'{dir_path}/data/nosid_data.csv',header=True)
        # sid_df.to_csv(f'{dir_path}/data/sid_data.csv',header=True)
        # clean_df.to_csv(f'{dir_path}/data/clean_df.csv',header=True)
        # clean_df = pd.read_csv(f'{dir_path}/data/clean_df.csv') 
    with st.echo():
        n_down=sum(clean_df['downloaded']==1) #76798
        n_notdown= sum(clean_df['downloaded']==0) #503054
        st.write(f'Downloaded tracks: {n_down}')
        st.write(f'Not downloaded tracks: {n_notdown}')
        import random
        random.seed(42)
        # Down-sampling
        df0 = clean_df[clean_df['downloaded']==0].sample(len(clean_df[clean_df['downloaded']==1]))
        balanced_df = pd.concat([clean_df[clean_df['downloaded']==1], df0], ignore_index=True)
        # Up-sampling
        # df1 = clean_df[clean_df['downloaded']==1].sample(len(clean_df[clean_df['downloaded']==0]), replace=True)
        # df_balanced = pd.concat([clean_df[clean_df['downloaded']==0], df1], ignore_index=True)
        # Sample specified number
        # n_samp=5000
        # balanced_df = pd.concat([clean_df[clean_df['downloaded']==0].sample(n_samp, replace=True), 
        #                         clean_df[clean_df['downloaded']==1].sample(n_samp, replace=True)], 
        #                         ignore_index=True)
        # balanced_df = balanced_df.drop('Unnamed: 0', axis=1)
        df=balanced_df
        st.write(f'Balanced data size: {df.shape}')
        # Save/Load balanced data
        # balanced_df.to_csv(f'{dir_path}/data/balanced_df.csv',header=True)
        # balanced_df = pd.read_csv(f'{dir_path}/data/balanced_df.csv')
# with st.echo():
# with st.echo():
    

## Tab: EDA -------------------------------------------------------
with tabs[1]:
    # st.dataframe(balanced_df.head())
    df = pd.read_csv(f'{dir_path}/balanced_df.csv') 
    with st.echo():
        import matplotlib.pyplot as plt
        import pandas as pd
        from textblob import TextBlob

        df = pd.read_csv(f'{data_path}/balanced_df.csv')
        df['sid'].value_counts()[df['sid'].value_counts()>1]
    
    with st.echo():
        # Calculate repeated sids and select those appearing more then once
        df_rep_sids = df['sid'].value_counts()[df['sid'].value_counts()>1]
        print(f'Top-10 most used sids:\n{df_rep_sids.iloc[:10]}\n')
        # Calculate repeated inputs and select those appearing more then once
        df_rep_inputs = df['input_norm'].value_counts()[df['input_norm'].value_counts()>1]
        print(f'Top-10 most used inputs:\n{df_rep_inputs.iloc[:10]}\n')
        df_rep = pd.DataFrame(df_rep_inputs[:20].values, index=df_rep_inputs[:20].keys())
        df_rep.plot.barh()
        plt.title('Repeated requests frequency (top 20)')
        plt.ylabel('Request')
        plt.xlabel('n')
        plt.gcf().set_size_inches(5,5)
        plt.show()
        # Count repeated pair 'input/sid'
        print('Most repeated pairs input/sid:')
        df[['input','sid']].value_counts()[:30]
    st.image(f'{data_path}/frequest.png')
    
    st.write('''Time Series Analysis: The number of song requests and downloads over each month''')
    with st.echo():
        df['date'] = pd.to_datetime(df['date']) 

        # Count song requests and downloads for each month
        monthly_counts = df.groupby(pd.Grouper(key='date', freq='M'))['downloaded'].agg(['count', 'sum'])
        plt.plot(monthly_counts.index, monthly_counts['sum']/monthly_counts['count'], label='Downloads/Requests')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Downloads/Requests ration by Month')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()
    st.image(f'{data_path}/ts_down_req.png')

# with st.echo():
# with st.echo():
# with st.echo():
# with st.echo():
# with st.echo():

        # st.write(f'Balanced data size: {balanced_df.shape}')
    
## Tab: Predict Downloads -----------------------------------------
with tabs[2]:
    st.title('Predict download from query')
    st.header('Prepare data for classifier')
    with st.echo():
        # Load preprocessed data
        df = pd.read_csv(f'{dir_path}/balanced_df.csv')  
        # Split data
        train_data, test_data = train_test_split(df, 
                                                    test_size = 0.1, 
                                                    random_state = 12, 
                                                    stratify=df['downloaded'])
        train_data, valid_data = train_test_split(train_data, 
                                                    train_size=0.8, 
                                                    random_state=12, 
                                                    stratify=train_data['downloaded'])
        # Reset indexes
        test_data.reset_index(drop=True, inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        valid_data.reset_index(drop=True, inplace=True)
    st.write(f'Training Data Shape: {train_data.shape}\nTest Data Shape: {test_data.shape}\nValidation Data: {valid_data.shape}')
    st.markdown('''- **lang_mod** is used for pretraining a language model, the vocabulary from it *lang_mod.train_ds.vocab* is shared with the classification model
- **class_mod** is used for training a text classification model with the help of transfer learning''')

    with st.echo():
        from fastai.text.all import *
        
        def run_model(arch=None):
            ## Language Model
            # Create a language model data loader
            dl_lang = TextDataLoaders.from_df(train_df=train_data, 
                                                valid_df=valid_data, 
                                                df=test_data,
                                                text_col='input',
                                                is_lm=True,
                                                seq_len=seq_len,
                                                bs=bs,
                                                path='')
            # Create a language model learner
            lang_learner = language_model_learner(lang_mod, 
                                                    arch=arch, 
                                                    pretrained = True, 
                                                    drop_mult=0.3)
            # Find learning rate for the language learner
            suggested_lr_lang = lang_learner.lr_find()[0]

            #Train the language learner model
            lang_learner.fit_one_cycle(1, suggested_lr_lang)

            #Validate
            val_loss = lang_learner.validate()[0]
            print(f"Validation Loss: {val_loss:.4f}")

            # Fine-tune the later layers of the base models using freeze_to
            lang_learner.freeze_to(-2)
            lang_learner.fit_one_cycle(1, slice(suggested_lr_lang/(2.6**4),suggested_lr_lang))

            # Save the language learner encoder
            lang_learner.save_encoder(f'fai_langlrn_enc_{arch}')

            ## Classification Model data loader
            dl_class = TextDataLoaders.from_df(train_df=train_data, 
                                                valid_df=valid_data, 
                                                df=test_data,
                                                text_col='input', 
                                                label_col='downloaded', 
                                                vocab=dl_lang.train_ds.vocab, 
                                                is_lm=False,
                                                seq_len=72,
                                                bs=32,
                                                path='')
            # Create a classification model learner
            class_learner = text_classifier_learner(dl_class, 
                                                    drop_mult=0.3, 
                                                    arch = arch, 
                                                    pretrained = True)

            class_learner.load_encoder(f'fai_langlrn_enc_{arch}')
            # Find the learning rate for the classifier
            suggested_lr_class = class_learner.lr_find()[0]

            #Train the classification learner model
            class_learner.fit_one_cycle(1, suggested_lr_class)

            val_loss = class_learner.validate()[0]
            print(f"Validation Loss: {val_loss:.4f}")

            # Fine-tune the later layers of the base models using freeze_to
            class_learner.freeze_to(-2)
            class_learner.fit_one_cycle(1, slice(suggested_lr_class/(2.6**4),suggested_lr_class))

            #Savethe language learner encoder
            class_learner.save(f'fai_clslrn_{arch}')

            return class_learner

    st.markdown('''
        ```
        bs=32
        seq_len=72
        model_AWD_LSTM = run_model(arch=AWD_LSTM)
        ```
        ''')
    st.image(f'{data_path}/res1.png')
    st.image(f'{data_path}/lr_loss_1.png')
    st.image(f'{data_path}/lr_loss_2.png')
    
## Tab: Sentiment Analysis ----------------------------------------
with tabs[3]:
    st.title('Query Sentiment Analysis')
    st.header('Prepare data')
    
    st.write('''Sentiment Analysis: 
        Analyze the sentiment of the 'input' column and see if there's a correlation 
        between the sentiment of the requested song and the likelihood of it being downloaded.''')
    with st.echo():
        # !pip install textblob
        from textblob import TextBlob
        import matplotlib.pyplot as plt

        df['input_norm'] = df['input_norm'].astype(str)

        # Extract sentiment polarity of the inputs
        def get_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity
        df['sentiment'] = df['input_norm'].apply(get_sentiment)
        # Group the data by the sentiment polarity score and calculate the mean downloaded count for each group
        grouped_df = df.groupby('sentiment')['downloaded'].mean()

        import numpy as np
        grouped_df = df.groupby('sentiment')['downloaded'].mean()

        x=np.array(grouped_df.index)
        y=np.array(grouped_df.values)
        fit2 = np.polyfit(x,y, deg=1)
        fit_fn2 = np.poly1d(fit2)
        yy2=fit_fn2(x)
        plt.scatter(x,grouped_df,s=1,marker='+')
        plt.plot(x.tolist(),yy2.tolist(),c='k')
        # sents = [round(key, 2) for key in grouped_df.keys().tolist()]
        # ticks_to_show = range(len(sents))[::10]
        # plt.xticks(ticks=ticks_to_show, labels=sents[::10])
        plt.title('Average Download Count by Input Sentiment Polarity')
        plt.xticks(rotation=45)
        plt.xlabel('Sentiment')
        plt.ylabel('Mean downloads')
    st.image(f'{data_path}/senti_down.png')

## Tab: Other -----------------------------------------------------
with tabs[4]:
    import altair as alt
    # import streamlit as st

    df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
    c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c',  
                                        color='c')
    st.altair_chart(c, width=-1)