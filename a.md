    # P1 web scrapping
    
    Practical 1 : Web Scrapping
    
    import requests
    from bs4 import BeautifulSoup
    
    r = requests.get("https://www.rjcollege.edu.in")
    soup = BeautifulSoup(r.content, "html.parser")
    print(soup.prettify())
    
    links = soup.find_all("a")
    for i in links:
        print(i.get("href"), i.string)
    
    ======================================================================
    Practical 2 :  Page Rank
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    D = nx.DiGraph()
    print(D)
    
    D.add_weighted_edges_from([('A', 'B', 1), ('A', 'C', 1), ('C', 'A', 1), ('B', 'C', 1)])
    
    pos = nx.spring_layout(D)
    nx.draw(D, pos=pos, with_labels=True)
    plt.show
    
    ======================================================================
    
    Prac 3 : Sentiment analsis
    
    
    import pandas as pd
    import torch 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv("full path of hotel_review.csv")
    print(df.head())
    
    neutral_range = {"low": 4, "high": 5}
    
    df["Sentiment"] = df["Rating"].apply(lambda rating: "neural" if neutral_range["low"] <= rating <=neural_range
    
    print(df.head())
    
    ======================================================================
    
    
    Prac 4 : Spam Classifer
    
    import pandas as pd
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    data = pd.read_csv("Full-Path-Of--mail_data.csv")
    data.head()
    
    encoder = LabelEncoder()
    data['Category'] = encoder.fit_transform(data['Category'])
    data.head()
    
    x = data['Message']
    y = data['Category']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=2)
    
    tfidf=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)
    
    model = LogisticRegression()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    accuracy_score(y_test,pred)
    
    
    ======================================================================
    Practical 5 :  Basic Web Crawler
    
    import requests
    from bs4 import BeautifulSoup
    
    r = requests.get("https://www.google.com")
    
    p = r.text
    
    soup = BeautifulSoup(p, "html.parser")
    
    for i in soup.find_all('a'):
        print(i.get('href'))
    
    ======================================================================
    
    Practical 6 : Text Mining
    
    import nltk
    from nltk.probability import FreqDist
    from nltk.tokenize import word_tokenize
    
    nltk.download('punkt')
    
    text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the easternside"
    
    token=word_tokenize(text)
    print(token)
    
    fdist = FreqDist(token)
    print(fdist)
    
    fdist1 = fdist.most_common(10)
    print(fdist1)
    
    ======================================================================
    Practical 7 : strmming
    
    import nltk
    nltk.download('wordnet')
    
    from nltk.stem import PorterStemmer
    pst = PorterStemmer()
    pst.stem("waiting")
    
    print('-'*20)
    print("PORTER STEMMER")
    print('-'*20)
    stm = ["Waiting","Waited","Waits"]
    for word in stm:
        print(word+":"+pst.stem(word))
    
    print('-'*20)
    print("LANCASTER STEMMER")
    print('-'*20)
    
    from nltk.stem import LancasterStemmer
    lst = LancasterStemmer()
    stm = ["giving","given","gave"]
    
    for word in stm:
        print(word+":"+lst.stem(word))
    
    print('-'*20)
    print("WORD NET LEMMATIZER")
    print('-'*20)
    
    from nltk.stem import WordNetLemmatizer
    lemma = WordNetLemmatizer()
    print("rocks :",lemma.lemmatize("rocks"))
    print("corpora :",lemma.lemmatize("corpora"))
    
    import nltk
    nltk.download('omw-1.4')
    
    print('-'*20)
    print("WORD NET LEMMATIZER")
    print('-'*20)
    from nltk.stem import WordNetLemmatizer
    lemma = WordNetLemmatizer()
    print("rocks :",lemma.lemmatize("rocks"))
    print("corpora :",lemma.lemmatize("corpora"))
    
    
    ======================================================================
    
    Practical 8 : Stop Words
    
    import nltk
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    
    import nltk
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    
    a = set(stopwords.words('english'))
    text = "Cristiano Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal"
    text1 = word_tokenize(text)
    print('-'*15)
    print("TOKENS")
    print('-'*15)
    print(text1)
    stopwords = [x for x in text1 if x not in a]
    print('-'*15)
    print("STOPWORDS")
    print('-'*15)
    print(stopwords)
    
    text = "vote to choose a particular man or a group (party) to represent them in parliament"
    tex = word_tokenize(text)
    print('-'*15)
    print("POS TAG")
    print('-'*15)
    for token in tex:
        print(nltk.pos_tag([token]))
    
