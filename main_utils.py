
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dash import Dash, dcc, html, Input, Output
import re
import nltk
import pandas as pd
import dash_cytoscape as cyto
from pytube import YouTube
from PIL import Image
from wordcloud import WordCloud
import base64
from io import BytesIO
import spacy
import es_core_news_md
import string

class make_graph:
    def __init__(self,content,language='eng'):
        self.content=content
        self.language=language
        
    
    def clean(self):
        self.content=self.content.strip()
        
        if self.language=='eng':
            cleaned=re.sub('[^A-Za-z0-9]+',' ',self.content)
            cleaned=self.content.translate(str.maketrans('','',string.punctuation))
            
            nltk.download('wordnet')
            stopwords_list=stopwords.words('english')
            wnl = WordNetLemmatizer()
            content_list=cleaned.split()
           
            content_list=[i.lower() for i in content_list]
            new_content_list=[wnl.lemmatize(w) for w in content_list]
        elif self.language=='spa':
            cleaned=self.content.translate(str.maketrans('','',string.punctuation))
           
           
            nlp = es_core_news_md.load()
            document=nlp(cleaned)
            print(document)
            new_content_list=[token.lemma_.lower() for token in document]
            with open('spanish_stop_words.txt','r') as sfile:
                c=sfile.readlines()
            stopwords_list=[i.replace('\n','') for i in c]
            sfile.close()
        
      
        
        
        
        final_content_list=[i for i in new_content_list if i not in stopwords_list]
        final_content_list=[i for i in final_content_list if i!=' ']
        print(final_content_list)
        return final_content_list


    def render_graph(self,final_content_list,style_param='random',max_words='all'):       
        
        
        
        df=pd.DataFrame({'words':final_content_list})
        count_df=df['words'].value_counts()
        if max_words=='all':
            print('todos')
            nodes=list(count_df.index)
        else:
            print('nums')
            nodes=list(count_df.index[:int(max_words)])

        #dict_nodes={w:i+1 for i,w in enumerate(nodes)}
        df=df[df['words'].isin(nodes)]
        #df['num']=df['words'].apply(lambda x: dict_nodes[x])
        df['idx']=[i for i in range(1,len(df)+1)]
        df=df.set_index('idx')
        nodes=[{'data':{'id':df.loc[i,'words'],'label':df.loc[i,'words']}} for i in df.index]
        edges=[]
        for i in df.index:
            if i+1<df.index[-1]:
                source=df.loc[i,'words']
                target=df.loc[i+1,'words']
                if source!=target:
                    edges.append({'data':{'source':source,'target':target}})
        complete_graph=nodes+edges

        graphobj=cyto.Cytoscape(id='grafo',
            layout={'name':f'{style_param}'},
            style={'width': '50%', 'height': '500px'},
            elements=complete_graph,
            stylesheet=[{
                'selector': 'label',             # as if selecting 'node' :/
                'style': {
                    'content': 'data(label)',    # not to loose label content
                    'color': 'black',
                    'background-color': '#537FE7',
                    'line-color':'light-gray'  # applies to node which will remain pink if selected :/
                 }
            }]
        )
        return graphobj
    def render_word_cloud(self,final_content_list):
        wordcloud = WordCloud(background_color='white').generate(' '.join(final_content_list))
        wc_img = wordcloud.to_image()
        with BytesIO() as buffer:
            wc_img.save(buffer, 'png')
            img2 = base64.b64encode(buffer.getvalue()).decode()
        return html.Img(src="data:image/png;base64," + img2)




def Download(link,folder):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(folder)
    except:
        print("An error has occurred")
    print("Download is completed successfully")
    return youtubeObject.default_filename,youtubeObject.title


