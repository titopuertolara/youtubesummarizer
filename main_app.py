from dash import Dash, dcc, html, Input, Output,ctx,State,ALL,dash_table
import os

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import whisper
from keybert import KeyBERT
from main_utils import make_graph,Download

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

server = app.server
model = whisper.load_model("small")
kw_model=KeyBERT()

options_list=['random','circle','breadthfirst','concentric']
app.layout = html.Div([
    html.H2('Youtube video summarizer'),
    html.H3('Powered by Whisper and KeyBert'),
    html.P('Processing depends of your cpu and video length'),
    
    html.Div([
        html.Div([dcc.Dropdown(id='language',placeholder='Video language',value='eng',style={'width':'200px'},options=[{'label':i,'value':j} for i,j in zip(['English','Spanish'],['eng','spa'])])],style={'display':'inline-block'}),
        html.Div([dcc.Input(id='youtube-link',type='text',placeholder='Youtube link',style={'width':'500px','margin-left':'1%','margin-top':'-2.7%','position':'absolute'})],style={'display':'inline-block'}),
        html.Div([html.Button('Go!',id='submit_btn',n_clicks=0)],style={'display':'inline-block','margin-left':'40%','position':'absolute','margin-top':'0%'})
    ]),
    
    dcc.Loading(id='load3',children=[html.Div(id='loading-model')],type='graph'),
    
    html.Div(id='network-options',children=[
        html.P('Write "all" for all words (could be slowly)'),
        
        html.Div(id='net-style-box',children=[
            html.H5('Graph style and # of words to process'),
            html.Div([dcc.Dropdown(id='net-style',options=[{'label':i,'value':i} for i in options_list], value='concentric')],style={'display':'inline-block','width':'40%'}),
            html.Div([dcc.Input(id='nwords',value=100,placeholder='Number of words')],style={'display':'inline-block','width':'20%','position':'absolute'})
        ])
    ]),
    html.H3(id='video-title',style={'color':'red'}),
    html.Br(),
    dcc.Loading(id='load2',children=[html.Div(id='network-div')],type='graph'),
    dcc.Loading(id='load1',children=[html.Div(id='wordcloud-div',style={'margin-left':'50%','margin-top':'-40%'})],type='graph'),
    html.Div(id='bert-div',style={'margin-left':'50%'}),
    dcc.Loading(id='load4',children=[html.Div(id='keywords-div',style={'margin-left':'50%'})],type='cube'),
    html.Div(id='text-whisper-area'),
    
    dcc.Store(id='video-text')
    
  
         
    
])

@app.callback(Output('video-text','data'),
              Output('network-options','style'),
              Output('loading-model','children'),
              Output('text-whisper-area','children'),
              Output('video-title','children'),
              
              [Input('submit_btn','n_clicks'),
              State('youtube-link','value')])
def model_forward(nclicks,link):
    video_name=''
    txt_result={'text':''}
    net_style={'display':'none'}
    text_area=''
    video_title=''
    #print(link)
    if 'submit_btn'==ctx.triggered_id:
        message='Loading..'
        if link is not None:
            video_name,video_title=Download(link,'videos')
            txt_result=model.transcribe(f'videos/{video_name}')
            net_style={'display':'block'}
            os.remove(f'videos/{video_name}')
            
            #graphobj=make_graph(txt_result)
            #final_content_list=graphobj.clean()
            #graph_result=graphobj.render_graph(final_content_list,max_words=100)
            text_area=[html.H5('Video text'),dcc.Textarea(id='whisper-result',disabled=True,value=txt_result['text'],style={'width':'100%','height':'400px'})]
            

    #result = model.transcribe()
    return txt_result,net_style,'',text_area,video_title


@app.callback(Output('network-div','children'),
              Output('wordcloud-div','children'),
              Output('bert-div','children'),
            
             [Input('net-style','value'),
              Input('video-text','data'),
              Input('nwords','value'),
              Input('language','value')])
def render_network(net_style,video_text,nwords,lang):
    #print(video_text['text'])
    
    
    graph_result=''
    wc_html_img=''
    bert_field=''
    try:
        graphobj=make_graph(video_text['text'],language=lang)
        final_content_list=graphobj.clean()
        graph_result=graphobj.render_graph(final_content_list,net_style,max_words=nwords)
        wc_html_img=graphobj.render_word_cloud(final_content_list)
        bert_field=[html.H5('# of keywords and lenght (KeyBert)'),
                    dcc.Input(id={"type":"bert-group","index":"bert-nwords"},type='text',placeholder='top n',value=5),
                    dcc.Input(id={"type":"bert-group","index":"bert-range"},type='text',placeholder='length',value=1)
                 ]

    except Exception as e:
        print(e)
    
    return graph_result,wc_html_img,bert_field
@app.callback(Output('keywords-div','children'),
              [
               Input('video-text','data'),
               Input({'type':'bert-group','index':ALL},'value')
               ])
def show_keywords(content,bert_params):
    key_datatable=''
    print(bert_params)
    if len(bert_params)>0:
        try:
            keys=kw_model.extract_keywords(docs=content['text'], keyphrase_ngram_range=(1,int(bert_params[1])),top_n=int(bert_params[0]))
            key_words=[i[0] for i in keys]
            scores=[i[1] for i in keys]
            df=pd.DataFrame({'Keywords':key_words,'Score':scores})
            key_datatable= dash_table.DataTable(
                                id='bertscores',
                                style_table={'height':'200px','overflowY':'auto','overflowX':'auto','width':'auto'},
                                style_header={'backgroundColor': '#393F56','fontWeight': 'bold','color':'white'},
                                style_cell={'textAlign':'left'},
                                export_format='xlsx',
                                page_size=7,
                                columns=[{'name':i,'id':i} for i in df.columns],
                                data=df.to_dict('records')
            )
            print(bert_params)
        except Exception as e:
            print(e)
            return "Something's wrong :("
        

    return key_datatable




if __name__ == '__main__':
    app.run_server(debug=True)