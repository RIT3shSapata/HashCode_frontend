import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    print(rec)

recommend1(adj_list,10)

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

rec1 = recommend(adj_list,10)
print(rec1)