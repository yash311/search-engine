from tkinter import *
from tkinter import ttk
import nltk
import numpy as np
import math
import heapq
import string

import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import preprocess

corpus=[]
wordfreq={}
word_idf_values = {}
word_tf_values = {}
tfidf_values = []
tfidf={}

documents = []


#reading the data
for i in range(1,16):
    f_name=""+str(i)+".txt"
    file=open(f_name,'r')
    temp_data=file.read().lower()
    documents.append(preprocess.preprocessor(temp_data))
    file.close()

# compute Term frequency of a specific term in a document
def termFrequency(term, document):
    normalizeDocument = document.split()
    return normalizeDocument.count(term) / float(len(normalizeDocument))
    
# IDF of a term
def inverseDocumentFrequency(term, documents):
    count = 0
    for doc in documents:
        if term in doc.split():
            count += 1
    if count > 0:
        return 1.0 + math.log(float(len(documents))/count)
    else:
        return 1.0

        
# tf-idf of a term in a document
def tf_idf(term, document, documents):
    tf = termFrequency(term, document)
    idf = inverseDocumentFrequency(term, documents)
    return tf*idf

def get_tf_idf(documents):
    tf_idf_scored_docs = []
    for doc in documents:
        doc_word_score = []
        for term in doc.split(" "):
            doc_word_score.append(tf_idf(term, doc, documents))
        tf_idf_scored_docs.append(dict(zip(doc.split(" "), doc_word_score)))
    return tf_idf_scored_docs

tf_idf_scored_docs = get_tf_idf(documents)

# print("^^^^^^^^^^^^^^^^^^^^^^^^ Printing tfidf score ^^^^^^^^^^^^^^^^^^^^^^^^^^")
# print(tf_idf_scored_docs[:3])
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")





def get_cosine_sim(tf_idf_scored_docs, tf_idf_scored_test_doc):
    doc_query_vector = []   # Vector that has gotten according to query words
    query_words = tf_idf_scored_test_doc.keys()
    query_score = list(tf_idf_scored_test_doc.values())

    for doc in tf_idf_scored_docs:
        vect = []
        doc_words = doc.keys()
        for query_word in query_words:
            if query_word in doc_words:
                vect.append(doc[query_word])
            else:
                vect.append(0)
        doc_query_vector.append(vect)


    print(query_score)
    doc_query_similarities = []
    for d in range(0, len(tf_idf_scored_docs)):
        sim = cosine_similarity([doc_query_vector[d]], [query_score])[0][0]
        doc_query_similarities.append((d, sim))
    
    return doc_query_similarities


    

''' Main function performing search operation '''
def search():
    
    # canvas.create_text(100, 100, anchor=W, font="Purisa",text="")
    canvas.delete("all")
    sw = preprocess.preprocessor(searchEntry.get().lower())

    print("LEN : +++++++++++++++++++++ ", str(len(tf_idf_scored_docs)))
    print("#################################################################")
    print(sw)

    test_document = sw.strip()


    print("****************************************************************")
    print(tf_idf_scored_docs[:3])
    tf_idf_scored_test_doc = dict(zip(test_document.split(" "),  [tf_idf(term, test_document, documents) for term in test_document.split(" ")]))

    # for term in test_document.split(" "):
    #     print(tf_idf(term, test_document, documents))

    arr = get_cosine_sim(tf_idf_scored_docs, tf_idf_scored_test_doc)
    arr.sort(key = lambda x: x[1], reverse=True)  
    # ans=[]
    # ans=heapq.nlargest(10,arr)
    
    print(arr)
    searched_doc = ""
    for ind,val in arr:
        id=ind+1
        searched_doc+=str(id)+"   ( "+str(val)+" )\n"
    canvas.create_text(200, 200, anchor=W, font="Purisa",text=searched_doc)
    

# Hogwarts School of Witchcraft and Wizardry and learns about magic




''' *********************************** UI ********************************************'''
root=Tk()
root.title("Gogole")
root.maxsize(900,600)
root.config(bg="black")


UI_frame = Frame(root, width=600, height=200, bg='grey')
UI_frame.grid(row=0, column=0, padx=10, pady=5)


canvas = Canvas(root, width=600, height=380, bg="white")
canvas.grid(row=1, column=0, padx=10, pady=5)


searchEntry=Entry(UI_frame)
searchEntry.grid(row=1, column=1, padx=5, pady=5, sticky=W)
Button(UI_frame, text='Click to search',command = search, bg='red').grid(row=1,column=2,padx=5,pady=5)

print("About to start")
root.mainloop()

