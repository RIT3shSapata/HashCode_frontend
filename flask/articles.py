import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urlparse
import pandas as pd
import _pickle as cPickle
from pathlib import Path
import spacy
from spacy import displacy
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

# glove_file = 'data/embeddings/glove.6B.300d.txt'
# tmp_file = 'data/embeddings/word2vec-glove.6B.300d.txt'

# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_file, tmp_file)
# model = KeyedVectors.load_word2vec_format(tmp_file)
import pickle
nlp = spacy.load('en_core_web_sm')

def googleSearch(query):
    g_clean = [ ]
    url = 'https://www.google.com/search?client=ubuntu&channel=fs&q={}&ie=utf-8&oe=utf-8'.format(query)
    try:
            html = requests.get(url)
            if html.status_code==200:
                soup = BeautifulSoup(html.text, 'lxml')
                a = soup.find_all('a') 
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

res = googleSearch("basic deep learning articles")

url = res[4]
html = requests.get(url)
article = ''
if html.status_code==200:
    soup = BeautifulSoup(html.text, 'lxml')
    for node in soup.findAll('p'):
        article = article + (''.join(node.findAll(text=True)))

def dumpPickle(fileName, content):
    pickleFile = open(fileName, 'wb')
    cPickle.dump(content, pickleFile, -1)
    pickleFile.close()

def loadPickle(fileName):    
    file = open(fileName, 'rb')
    content = cPickle.load(file)
    file.close()
    
    return content
    
def pickleExists(fileName):
    file = Path(fileName)
    
    if file.is_file():
        return True
    
    return False

#Extract answers and the sentence they are in
def extractAnswers(qas, doc):
    answers = []

    senStart = 0
    senId = 0

    for sentence in doc.sents:
        senLen = len(sentence.text)

        for answer in qas:
            answerStart = answer['answers'][0]['answer_start']

            if (answerStart >= senStart and answerStart < (senStart + senLen)):
                answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

        senStart += senLen
        senId += 1
    
    return answers

def tokenIsAnswer(token, sentenceId, answers):
    for i in range(len(answers)):
        if (answers[i]['sentenceId'] == sentenceId):
            if (answers[i]['text'] == token):
                return True
    return False

#Save named entities start points

def getNEStartIndexs(doc):
    neStarts = {}
    for ne in doc.ents:
        neStarts[ne.start] = ne
        
    return neStarts 

def getSentenceStartIndexes(doc):
    senStarts = []
    
    for sentence in doc.sents:
        senStarts.append(sentence[0].i)
    
    return senStarts
    
def getSentenceForWordPosition(wordPos, senStarts):
    for i in range(1, len(senStarts)):
        if (wordPos < senStarts[i]):
            return i - 1

def addWordsForParagrapgh(newWords, text):
    doc = nlp(text)

    neStarts = getNEStartIndexs(doc)
    senStarts = getSentenceStartIndexes(doc)
    
    #index of word in spacy doc text
    i = 0
    
    while (i < len(doc)):
        #If the token is a start of a Named Entity, add it and push to index to end of the NE
        if (i in neStarts):
            word = neStarts[i]
            #add word
            currentSentence = getSentenceForWordPosition(word.start, senStarts)
            wordLen = word.end - word.start
            shape = ''
            for wordIndex in range(word.start, word.end):
                shape += (' ' + doc[wordIndex].shape_)

            newWords.append([word.text,
                            0,
                            0,
                            currentSentence,
                            wordLen,
                            word.label_,
                            None,
                            None,
                            None,
                            shape])
            i = neStarts[i].end - 1
        #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
        else:
            if (doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]

                currentSentence = getSentenceForWordPosition(i, senStarts)
                wordLen = 1

                newWords.append([word.text,
                                0,
                                0,
                                currentSentence,
                                wordLen,
                                None,
                                word.pos_,
                                word.tag_,
                                word.dep_,
                                word.shape_])
        i += 1

def oneHotEncodeColumns(df):
    columnsToEncode = ['NER', 'POS', "TAG", 'DEP']

    for column in columnsToEncode:
        one_hot = pd.get_dummies(df[column])
        one_hot = one_hot.add_prefix(column + '_')

        df = df.drop(column, axis = 1)
        df = df.join(one_hot)
    
    return df

def generateDf(text):
    words = []
    addWordsForParagrapgh(words, text)

    wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
    df = pd.DataFrame(words, columns=wordColums)
    
    return df

def prepareDf(df):
    #One-hot encoding
    wordsDf = oneHotEncodeColumns(df)

    #Drop unused columns
    columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape']
    wordsDf = wordsDf.drop(columnsToDrop, axis = 1)

    #Add missing colums 
    predictorColumns = ['wordCount','NER_CARDINAL','NER_DATE','NER_EVENT','NER_FAC','NER_GPE','NER_LANGUAGE','NER_LAW','NER_LOC','NER_MONEY','NER_NORP','NER_ORDINAL','NER_ORG','NER_PERCENT','NER_PERSON','NER_PRODUCT','NER_QUANTITY','NER_TIME','NER_WORK_OF_ART','POS_ADJ','POS_ADP','POS_ADV','POS_CCONJ','POS_DET','POS_INTJ','POS_NOUN','POS_NUM','POS_PART','POS_PRON','POS_PROPN','POS_PUNCT','POS_SYM','POS_VERB','POS_X','TAG_''','TAG_-LRB-','TAG_.','TAG_ADD','TAG_AFX','TAG_CC','TAG_CD','TAG_DT','TAG_EX','TAG_FW','TAG_IN','TAG_JJ','TAG_JJR','TAG_JJS','TAG_LS','TAG_MD','TAG_NFP','TAG_NN','TAG_NNP','TAG_NNPS','TAG_NNS','TAG_PDT','TAG_POS','TAG_PRP','TAG_PRP$','TAG_RB','TAG_RBR','TAG_RBS','TAG_RP','TAG_SYM','TAG_TO','TAG_UH','TAG_VB','TAG_VBD','TAG_VBG','TAG_VBN','TAG_VBP','TAG_VBZ','TAG_WDT','TAG_WP','TAG_WRB','TAG_XX','DEP_ROOT','DEP_acl','DEP_acomp','DEP_advcl','DEP_advmod','DEP_agent','DEP_amod','DEP_appos','DEP_attr','DEP_aux','DEP_auxpass','DEP_case','DEP_cc','DEP_ccomp','DEP_compound','DEP_conj','DEP_csubj','DEP_csubjpass','DEP_dative','DEP_dep','DEP_det','DEP_dobj','DEP_expl','DEP_intj','DEP_mark','DEP_meta','DEP_neg','DEP_nmod','DEP_npadvmod','DEP_nsubj','DEP_nsubjpass','DEP_nummod','DEP_oprd','DEP_parataxis','DEP_pcomp','DEP_pobj','DEP_poss','DEP_preconj','DEP_predet','DEP_prep','DEP_prt','DEP_punct','DEP_quantmod','DEP_relcl','DEP_xcomp']

    for feature in predictorColumns:
        if feature not in wordsDf.columns:
            wordsDf[feature] = 0
    
    return wordsDf

def predictWords(wordsDf, df):
    
    predictorPickleName = 'data/pickles/nb-predictor.pkl'
    predictor = loadPickle(predictorPickleName)
    
    y_pred = predictor.predict_proba(wordsDf)

    labeledAnswers = []
    for i in range(len(y_pred)):
        labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})
    
    return labeledAnswers

def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
    leftPartStart = doc[sentStart].idx
    leftPartEnd = doc[firstTokenIndex].idx
    rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex])
    rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
    
    question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]
    
    return question

def addQuestions(answers, text):
    doc = nlp(text)
    currAnswerIndex = 0
    qaPair = []

    #Check wheter each token is the next answer
    for sent in doc.sents:
        for token in sent:
            
            #If all the answers have been found, stop looking
            if currAnswerIndex >= len(answers):
                break
            
            #In the case where the answer is consisted of more than one token, check the following tokens as well.
            answerDoc = nlp(answers[currAnswerIndex]['word'])
            answerIsFound = True
            
            for j in range(len(answerDoc)):
                if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
                    answerIsFound = False
           
            #If the current token is corresponding with the answer, add it 
            if answerIsFound:
                question = blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)
                
                qaPair.append({'question' : question, 'answer': answers[currAnswerIndex]['word'], 'prob': answers[currAnswerIndex]['prob']})
                
                currAnswerIndex += 1
                
    return qaPair

def sortAnswers(qaPairs):
    orderedQaPairs = sorted(qaPairs, key=lambda qaPair: qaPair['prob'])
    
    return orderedQaPairs    

f = open('model.pkl','rb')
model = pickle.load(f)
f.close()
# glove_file = 'data/embeddings/glove.6B.300d.txt'
# tmp_file = 'data/embeddings/word2vec-glove.6B.300d.txt'
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
def generate_distractors(answer, count):
    answer = str.lower(answer)
    
    ##Extracting closest words for the answer. 
    try:
        closestWords = model.most_similar(positive=[answer], topn=count)
    except:
        #In case the word is not in the vocabulary, or other problem not loading embeddings
        return []

    #Return count many distractors
    distractors = list(map(lambda x: x[0], closestWords))[0:count]
    
    return distractors

def addDistractors(qaPairs, count):
    for qaPair in qaPairs:
        distractors = generate_distractors(qaPair['answer'], count)
        qaPair['distractors'] = distractors
    
    return qaPairs

def generateQuestions(text, count):
    
    # Extract words 
    df = generateDf(text)
    wordsDf = prepareDf(df)
    
    # Predict 
    labeledAnswers = predictWords(wordsDf, df)
    
    # Transform questions
    qaPairs = addQuestions(labeledAnswers, text)
    
    # Pick the best questions
    orderedQaPairs = sortAnswers(qaPairs)
    
    # Generate distractors
    questions = addDistractors(orderedQaPairs[:count], 4)
    
    print(sorted(questions, key = lambda i: i['prob'],reverse=True))

# text = 'This article is part of the Complete Beginner’s Guide to Deep Learning series.It’s learning from examples. That’s pretty much the deal!At a very basic level, deep learning is a machine learning technique. It teaches a computer to filter inputs through layers to learn how to predict and classify information. Observations can be in the form of images, text, or sound.The inspiration for deep learning is the way that the human brain filters information. Its purpose is to mimic how the human brain works to create some real magic.Deep learning attempts to mimic the activity in layers of neurons in the neocortex.It’s very literally an artificial neural network.In the human brain, there are about 100 billion neurons. Each neuron connects to about 100,000 of its neighbors. That is what we’re trying to create, but in a way and at a level that works for machines.What does this mean in terms of neurons, axons, dendrites, and so on? Well, the neuron has a body, dendrites, and an axon. The signal from one neuron travels down the axon and transfers to the dendrites of the next neuron. That connection where the signal passes is called a synapse.Neurons by themselves are kind of useless. But when you have lots of them, they work together to create some serious magic. That’s the idea behind a deep learning algorithm! You get input from observation and you put your input into one layer. That layer creates an output which in turn becomes the input for the next layer, and so on. This happens over and over until your final output signal!So the neuron (or node) gets a signal or signals (input values), which pass through the neuron. That neuron delivers the output signal. Think of the input layer as your senses: the things you, for example, see, smell, and feel. These are independent variables for one single observation. This information is broken down into numbers and the bits of binary data that a computer can use. (You will need to either standardize or normalize these variables so that they’re within the same range.)What about synapses? Each of the synapses gets assigned weights, which are crucial to Artificial Neural Networks (ANNs). Weights are how ANNs learn. By adjusting the weights, the ANN decides to what extent signals get passed along. When you’re training your network, you’re deciding how the weights are adjusted.There are two different approaches to get a program to do what you want. First, there’s the specifically guided and hard-programmed approach. In this approach, you tell the program exactly what you want it to do. Then there are neural networks. In neural networks, you tell your network the inputs and what you want for the outputs, and let it learn on its own. By allowing the network to learn on its own, we can avoid the necessity of entering in all the rules. For a neural network, you can create the architecture and then let it go and learn. Once it’s trained up, you can give it a new image and it will be able to distinguish output.There are different kinds of neural networks. They’re generally classified into feedforward and feedback networks.A feedforward network is a network that contains inputs, outputs, and hidden layers. The signals can only travel in one direction (forward). Input data passes into a layer where calculations are performed. Each processing element computes based upon the weighted sum of its inputs. The new values become the new input values that feed the next layer (feed-forward). This continues through all the layers and determines the output. Feedforward networks are often used in, for example, data mining.A feedback network (for example, a recurrent neural network) has feedback paths. This means that they can have signals traveling in both directions using loops. All possible connections between neurons are allowed. Since loops are present in this type of network, it becomes a non-linear dynamic system which changes continuously until it reaches a state of equilibrium. Feedback networks are often used in optimization problems where the network looks for the best arrangement of interconnected factors.The majority of modern deep learning architectures are based on artificial neural networks (ANNs). They use many layers of nonlinear processing units for feature extraction and transformation. Each successive layer uses the output of the previous layer for its input. What they learn forms a hierarchy of concepts. In this hierarchy, each level learns to transform its input data into a more and more abstract and composite representation.That means that for an image, for example, the input might be a matrix of pixels. The first layer might encode the edges and compose the pixels. The next layer might compose an arrangement of edges. The next layer might encode a nose and eyes. The next layer might recognize that the image contains a face, and so on.What happens inside the neuron? The input node takes in information that in a numerical form. The information is presented as an activation value where each node is given a number. The higher the number, the greater the activation.Based on the connection strength (weights) and transfer function, the activation value passes to the next node. Each of the nodes sums the activation values that it receives (it calculates the weighted sum) and modifies that sum based on its transfer function. Next, it applies an activation function. An activation function is a function that’s applied to this particular neuron. From that, the neuron understands if it needs to pass along a signal or not. The activation runs through the network until it reaches the output nodes. The output nodes then give us the information in a way that we can understand. Your network will use a cost function to compare the output and the actual expected output. The model performance is evaluated by the cost function. It’s expressed as the difference between the actual value and the predicted value. There are many different cost functions you can use, you’re looking at what the error you have in your network is. You’re working to minimize loss function. (In essence, the lower the loss function, the closer it is to your desired output). The information goes back, and the neural network begins to learn with the goal of minimizing the cost function by tweaking the weights. This process is called backpropagation.Interested in learning more about cost functions? Check out A List of Cost Functions Used in Neural Networks, Alongside Applications on Stack ExchangeIn forward propagation, information is entered into the input layer and propagates forward through the network to get our output values. We compare the values to our expected results. Next, we calculate the errors and propagate the info backward. This allows us to train the network and update the weights. Backpropagation allows us to adjust all the weights simultaneously. During this process, because of the way the algorithm is structured, you’re able to adjust all of the weights simultaneously. This allows you to see which part of the error each of your weights in the neural network is responsible for.Hungry for more? You might want to read Efficient BackProp by Yann LeCun, et al., as well as Neural Networks and Deep Learning by Michael Nielsen.When you’ve adjusted the weights to the optimal level, you’re ready to proceed to the testing phase!Inputs to a neuron can either be features from a training set or outputs from the neurons of a previous layer. Each connection between two neurons has a unique synapse with a unique weight attached. If you want to get from one neuron to the next, you have to travel along the synapse and pay the “toll” (weight). The neuron then applies an activation function to the sum of the weighted inputs from each incoming synapse. It passes the result on to all the neurons in the next layer. When we talk about updating weights in a network, we’re talking about adjusting the weights on these synapses.A neuron’s input is the sum of weighted outputs from all the neurons in the previous layer. Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron. If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights: one for each synapse.So what is an activation function?In a nutshell, the activation function of a node defines the output of that node.The activation function (or transfer function) translates the input signals to output signals. It maps the output values on a range like 0 to 1 or -1 to 1. It’s an abstraction that represents the rate of action potential firing in the cell. It’s a number that represents the likelihood that the cell will fire. At it’s simplest, the function is binary: yes (the neuron fires) or no (the neuron doesn’t fire). The output can be either 0 or 1 (on/off or yes/no), or it can be anywhere in a range. If you were using a function that maps a range between 0 and 1 to determine the likelihood that an image is a cat, for example, an output of 0.9 would show a 90% probability that your image is, in fact, a cat.What options do we have? There are many activation functions, but these are the four very common ones:Want to dive deeper? Check out Deep Sparse Rectifier Neural Networks by Xavier Glorot, et al.So let’s say, for example, your desired value is binary. You’re looking for a “yes” or a “no.” Which activation function do you want to use? From the above examples, you could use the threshold function, or you could go with the sigmoid activation function. The sigmoid function would be able to give you the probability of a yes.So, how are the weights adjusted, exactly?You could use a brute force approach to adjust the weights and test thousands of different combinations. Even with the most simple neural network that has only five input values and a single hidden layer, you’ll wind up with 10⁷⁵ possible combinations. Running this on the world’s fastest supercomputer '#would take longer than the universe has existed so far.However, if you go with gradient descent, you can look at the angle of the slope of the weights and find out if it’s positive or negative in order to continue to slope downhill to find the best weights on your quest to reach the global minimum.If you go with gradient descent, you can look at the angle of the slope of the weights and find out if it’s positive or negative. This allows you to continue to slope downhill to find the best weights on your quest to reach the global minimum.Gradient descent is an algorithm for finding the minimum of a function. The analogy you’ll see over and over is that of someone stuck on top of a mountain and trying to get down (find the minima). There’s heavy fog making it impossible to see the path, so she uses gradient descent to get down to the bottom of the mountain. She looks at the steepness of the hill where she is and proceeds down in the direction of the steepest descent. You should assume that the steepness isn’t immediately obvious. Luckily she has a tool that can measure steepness. Unfortunately, this tool takes forever. She wants to use it as infrequently as she can to get down the mountain before dark. The real difficulty is choosing how often she wants to use her tool so she doesn’t go off track. In this analogy, the person is the algorithm. The steepness of the hill is the slope of the error surface at that point. The direction she goes is the gradient of the error surface at that point. The tool she’s using is differentiation (the slope of the error surface can be calculated by taking the derivative of the squared error function at that point). The rate at which she travels before taking another measurement is the learning rate of the algorithm. It’s not a perfect analogy, but it gives you a good sense of what gradient descent is all about. The machine is learning the gradient, or direction, that the model should take to reduce errors.Stochastic Gradient DescentGradient descent requires the cost function to be convex, but what if it isn’t?Normal gradient descent will get stuck at a local minimum rather than a global minimum, resulting in a subpar network. In normal gradient descent, we take all our rows and plug them into the same neural network, take a look at the weights, and then adjust them. This is called batch gradient descent. In stochastic gradient descent, we take the rows one by one, run the neural network, look at the cost functions, adjust the weights, and then move to the next row. Essentially, you’re adjusting the weights for each row.Stochastic gradient descent has much higher fluctuations, which allows you to find the global minimum. It’s called “stochastic” because samples are shuffled randomly, instead of as a single group or as they appear in the training set. It looks like it might be slower, but it’s actually faster because it doesn’t have to load all the data into memory and wait while the data is all run together. The main pro for batch gradient descent is that it’s a deterministic algorithm. This means that if you have the same starting weights, every time you run the network you will get the same results. Stochastic gradient descent is always working at random. (You can also run mini-batch gradient descent where you set a number of rows, run that many rows at a time, and then update your weights.)Many improvements on the basic stochastic gradient descent algorithm have been proposed and used, including implicit updates (ISGD), momentum method, averaged stochastic gradient descent, adaptive gradient algorithm (AdaGrad), root mean square propagation (RMSProp), adaptive moment estimation (Adam), and more.Loving this? You might want to take a look at A Neural Network in 13 lines of Python-Part 2 Gradient Descent by Andrew Trask and Neural Networks and Deep Learning by Michael NielsenSo here’s a quick walkthrough of training an artificial neural network with stochastic gradient descent:That’s it! You now know the basic ideas behind what’s happening in an artificial neural network!Still with me? Come on over to part 3!As always, if you do anything cool with this information, leave a comment in the responses below or reach out any time on LinkedIn @annebonnerdata!Written byHands-on real-world examples, research,  tutorials, and cutting-edge techniques delivered Monday to Thursday. Make learning your daily ritual. \xa0Take a look'
# text = """This article is part of the Complete Beginner’s Guide to Deep Learning series.It’s learning from examples. That’s pretty much the deal!At a very basic level, deep learning is a machine learning technique. It teaches a computer to filter inputs through layers to learn how to predict and classify information. Observations can be in the form of images, text, or sound.The inspiration for deep learning is the way that the human brain filters information. Its purpose is to mimic how the human brain works to create some real magic.Deep learning attempts to mimic the activity in layers of neurons in the neocortex.It’s very literally an artificial neural network.In the human brain, there are about 100 billion neurons. Each neuron connects to about 100,000 of its neighbors. That is what we’re trying to create, but in a way and at a level that works for machines.What does this mean in terms of neurons, axons, dendrites, and so on? Well, the neuron has a body, dendrites, and an axon. The signal from one neuron travels down the axon and transfers to the dendrites of the next neuron. That connection where the signal passes is called a synapse.Neurons by themselves are kind of useless. But when you have lots of them, they work together to create some serious magic. That’s the idea behind a deep learning algorithm! You get input from observation and you put your input into one layer. That layer creates an output which in turn becomes the input for the next layer, and so on. This happens over and over until your final output signal!So the neuron (or node) gets a signal or signals (input values), which pass through the neuron. That neuron delivers the output signal. Think of the input layer as your senses: the things you, for example, see, smell, and feel. These are independent variables for one single observation. This information is broken down into numbers and the bits of binary data that a computer can use. (You will need to either standardize or normalize these variables so that they’re within the same range.)What about synapses? Each of the synapses gets assigned weights, which are crucial to Artificial Neural Networks (ANNs). Weights are how ANNs learn. By adjusting the weights, the ANN decides to what extent signals get passed along. When you’re training your network, you’re deciding how the weights are adjusted.There are two different approaches to get a program to do what you want. First, there’s the specifically guided and hard-programmed approach. In this approach, you tell the program exactly what you want it to do. Then there are neural networks. In neural networks, you tell your network the inputs and what you want for the outputs, and let it learn on its own. By allowing the network to learn on its own, we can avoid the necessity of entering in all the rules. For a neural network, you can create the architecture and then let it go and learn. Once it’s trained up, you can give it a new image and it will be able to distinguish output.There are different kinds of neural networks. They’re generally classified into feedforward and feedback networks.A feedforward network is a network that contains inputs, outputs, and hidden layers. The signals can only travel in one direction (forward). Input data passes into a layer where calculations are performed. Each processing element computes based upon the weighted sum of its inputs. The new values become the new input values that feed the next layer (feed-forward). This continues through all the layers and determines the output. Feedforward networks are often used in, for example, data mining.A feedback network (for example, a recurrent neural network) has feedback paths. This means that they can have signals traveling in both directions using loops. All possible connections between neurons are allowed. Since loops are present in this type of network, it becomes a non-linear dynamic system which changes continuously until it reaches a state of equilibrium. Feedback networks are often used in optimization problems where the network looks for the best arrangement of interconnected factors.The majority of modern deep learning architectures are based on artificial neural networks (ANNs). They use many layers of nonlinear processing units for feature extraction and transformation. Each successive layer uses the output of the previous layer for its input. What they learn forms a hierarchy of concepts. In this hierarchy, each level learns to transform its input data into a more and more abstract and composite representation.That means that for an image, for example, the input might be a matrix of pixels. The first layer might encode the edges and compose the pixels. The next layer might compose an arrangement of edges. The next layer might encode a nose and eyes. The next layer might recognize that the image contains a face, and so on.What happens inside the neuron? The input node takes in information that in a numerical form. The information is presented as an activation value where each node is given a number. The higher the number, the greater the activation.Based on the connection strength (weights) and transfer function, the activation value passes to the next node. Each of the nodes sums the activation values that it receives (it calculates the weighted sum) and modifies that sum based on its transfer function. Next, it applies an activation function. An activation function is a function that’s applied to this particular neuron. From that, the neuron understands if it needs to pass along a signal or not. The activation runs through the network until it reaches the output nodes. The output nodes then give us the information in a way that we can understand. Your network will use a cost function to compare the output and the actual expected output. The model performance is evaluated by the cost function. It’s expressed as the difference between the actual value and the predicted value. There are many different cost functions you can use, you’re looking at what the error you have in your network is. You’re working to minimize loss function. (In essence, the lower the loss function, the closer it is to your desired output). The information goes back, and the neural network begins to learn with the goal of minimizing the cost function by tweaking the weights. This process is called backpropagation.Interested in learning more about cost functions? Check out A List of Cost Functions Used in Neural Networks, Alongside Applications on Stack ExchangeIn forward propagation, information is entered into the input layer and propagates forward through the network to get our output values. We compare the values to our expected results. Next, we calculate the errors and propagate the info backward. This allows us to train the network and update the weights. Backpropagation allows us to adjust all the weights simultaneously. During this process, because of the way the algorithm is structured, you’re able to adjust all of the weights simultaneously. This allows you to see which part of the error each of your weights in the neural network is responsible for.Hungry for more? You might want to read Efficient BackProp by Yann LeCun, et al., as well as Neural Networks and Deep Learning by Michael Nielsen.When you’ve adjusted the weights to the optimal level, you’re ready to proceed to the testing phase!Inputs to a neuron can either be features from a training set or outputs from the neurons of a previous layer. Each connection between two neurons has a unique synapse with a unique weight attached. If you want to get from one neuron to the next, you have to travel along the synapse and pay the “toll” (weight). The neuron then applies an activation function to the sum of the weighted inputs from each incoming synapse. It passes the result on to all the neurons in the next layer. When we talk about updating weights in a network, we’re talking about adjusting the weights on these synapses.A neuron’s input is the sum of weighted outputs from all the neurons in the previous layer. Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron. If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights: one for each synapse.So what is an activation function?In a nutshell, the activation function of a node defines the output of that node.The activation function (or transfer function) translates the input signals to output signals. It maps the output values on a range like 0 to 1 or -1 to 1. It’s an abstraction that represents the rate of action potential firing in the cell. It’s a number that represents the likelihood that the cell will fire. At it’s simplest, the function is binary: yes (the neuron fires) or no (the neuron doesn’t fire). The output can be either 0 or 1 (on/off or yes/no), or it can be anywhere in a range. If you were using a function that maps a range between 0 and 1 to determine the likelihood that an image is a cat, for example, an output of 0.9 would show a 90% probability that your image is, in fact, a cat.What options do we have? There are many activation functions, but these are the four very common ones:Want to dive deeper? Check out Deep Sparse Rectifier Neural Networks by Xavier Glorot, et al.So let’s say, for example, your desired value is binary. You’re looking for a “yes” or a “no.” Which activation function do you want to use? From the above examples, you could use the threshold function, or you could go with the sigmoid activation function. The sigmoid function would be able to give you the probability of a yes.So, how are the weights adjusted, exactly?You could use a brute force approach to adjust the weights and test thousands of different combinations. Even with the most simple neural network that has only five input values and a single hidden layer, you’ll wind up with 10⁷⁵ possible combinations. """
generateQuestions(article, 10)
