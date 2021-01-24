import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
from flask_cors import CORS, cross_origin
from flask import Flask,request,json
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urlparse
import pandas as pd
import _pickle as cPickle
from pathlib import Path
app = Flask(__name__)

cors = CORS(app)

f = open('graph_connections.txt','r')
connections = f.read()
f.close()

connections = connections.split('\n')

f = open('users.txt','r')
no_of_users = len(f.read().split('\n'))
f.close()

# cosine similarity
final_mat = []
df = pd.read_csv('friends.csv')

for column in df.columns:
    df[column]=df[column].apply(str)

def combined_features(row):
    return(row['experience']+" "+row['interest-1']+" "+row['interest-2']+" "+row['interest-3']+" "+row['linkedIn']+" "+row['github']+" "+row['proficiency-1']+" "+row['proficiency-2'])

users = df['name'].tolist()

for user in users:
    userfeatures = list(np.array(df[df['name']==user])[0])[1:]

    df['combined_features']=df.apply(combined_features,axis=1)

    count_matrix=CountVectorizer()
    count_matrix=count_matrix.fit_transform(df['combined_features'])

    cosine_sim_mat=cosine_similarity(count_matrix)

    friends = cosine_sim_mat[df.index[df['name']==user]][0].tolist()
    friends_list = []
    for i,j in enumerate(friends):
        friends_list.append((i,j))
        
    final_mat.append(friends_list)

# print(final_mat)

adj_list = [[] for i in range(no_of_users)]

for connection in connections:
    source,destination = connection.split('|')
    adj_list[int(source)-1].append((int(destination)-1,final_mat[int(source)-1][int(destination)-1][1]))
# print(adj_list)

def bfs(v,adj_list,visited,queue):
    visited[v] = 1
    # print(v)
    rec = []
    queue.append(v)
    while(len(queue)!=0):
        x = queue.pop(0)
        for i in adj_list[x]:
            if(visited[i[0]]==0):
                # print(i[0])
                queue.append(i[0])
                visited[i[0]] = 1
                if(i[1] > 0.6):
                    rec.append(i[0])
    return rec

def recommend1(adj_list,v):
    queue = []
    visited = [0]*no_of_users
    rec = bfs(v,adj_list,visited,queue)
    return(rec)


def googleSearch(query):
    g_clean = [ ] #this is the list we store the search results
    url = 'https://www.google.com/search?client=ubuntu&channel=fs&q={}&ie=utf-8&oe=utf-8'.format(query)#this is the actual query we are going to scrape
    try:
            html = requests.get(url)
            if html.status_code==200:
                soup = BeautifulSoup(html.text, 'lxml')
                a = soup.find_all('a') # a is a list
                for i in a:
                    k = i.get('href')
                    try:
                        m = re.search("(?P<url>https?://[^\s]+)", k)
                        n = m.group(0)
                        rul = n.split('&')[0]
                        domain = urlparse(rul)
                        if(re.search('google.com', domain.netloc)):
                            continue
                        else:
                            g_clean.append(rul)
                    except:
                        continue
    except Exception as ex:
            print(str(ex))
    finally:
            return g_clean


def recommend(g,v):
    friends = []
    for friend in g[v]:
        if(friend[1] > 0.3 and friend[0] not in friends):
            friends.append(friend[0])
        for friend2 in g[friend[0]]:
            if(friend2[1] > 0.3 and friend2[0] not in friends):
                friends.append(friend2[0])
                for friend3 in g[friend2[0]]:
                    if(friend3[1] > 0.3 and friend3[0] not in friends):
                        friends.append(friend3[0])

    return friends

# def get_articles(query)
def get_exp():
    df = pd.read_csv('friends.csv')
    xi = list(df.iloc[10])
    exp = [df.iloc[x] for x in range(len(df)) if df.iloc[x]['experience']>10]
    exp1 = [x['index'] for x in exp if x['interest-1'] in xi]
    exp2 = [x['index'] for x in exp if x['interest-2'] in xi]
    exp3 = [x['index'] for x in exp if x['interest-3'] in xi]
    # print(type(exp),type(exp1),type(exp2))
    # print(xi,type(xi))
    return(xi,exp1[:5],exp2[:5],exp3[:5])
# get_exp()
@app.route('/recommend')
@cross_origin()
def get_recommend():
    top_rec = recommend(adj_list,10)
    top_rec2 = recommend1(adj_list,10)
    exp1,exp2,exp3,x = get_exp()[1],get_exp()[2],get_exp()[3],get_exp()[0]
    return {'friends_rec':top_rec,'rec':top_rec2}

@app.route('/articles',methods=["POST"])
def foo():
    data = request.json
    # que = []
    try:
        print(data['title'])
        arts = googleSearch(data['title'])
        # for i in arts:
        #     html = requests.get(i)
        #     article = ''
        #     if html.status_code==200:
        #         soup = BeautifulSoup(html.text, 'lxml')
        #         for node in soup.findAll('p'):
        #             article = article + (''.join(node.findAll(text=True)))
                    # print(article)
                    # try:
                        # print(generateQuestions(article,10))
                        # que.append(generateQuestions(article,10))
                    # except:
                        # que.append('no questions')
        print(arts)
        return {'arts':arts}
    except:
        return {'arts':'nope'}

# @app.route('/updateFile',methods=["POST"])
# def foo5():
#     data = request.json
#     try:
#         print(data['title'])
#         maxp,tt,user = int(data['title'].split('|')[1]),int(data['title'].split('|')[0]),int(data['title'].split('|')[2])
#         f= open('score.txt')
#         txt = f.read()#.split('\n')
#         f.close()
#         print(txt)
#         score = []
#         for i in txt:
#             score.append((i.split(',')[0],i.split(',')[1]))
#         print(score)
#         for i in range(len(score)):
#             if(tt > score[i][1]):
#                 maxp = (maxp - (score[i][1]/(tt - score[i][1])))
#             elif(tt<score[i][1]):
#                 maxp = (maxp + (score[i]/(tt - score[i][1])))

#         print(maxp)
#         f = open('score.txt','w')
#         writes = ""
#         for i in range(len(score)):
#             if(i==user):
#                 writes = writes + str(maxp)+','+str(tt)+'\n'
#             else:
#                 writes = writes + score[i][0]+','+score[i][1]+'\n'
#         print("written",writes)
#         f.write(writes)
#         f.close()

#         print(txt)


#         print(maxp,tt)
#         return {'arts':maxp}
#     except:
#         return {'arts':'nope'}

