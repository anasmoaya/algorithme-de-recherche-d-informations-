import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def create_dictionary(tokens_list):
    dictionary = set()
    for tokens in tokens_list:
        dictionary.update(tokens)
    return sorted(list(dictionary))

def build_incidence_matrix(documents, dictionary):
    matrix = np.zeros((len(documents), len(dictionary)), dtype=int)
    for doc_idx, doc in enumerate(documents):
        for word in doc:
            if word in dictionary:
                word_idx = dictionary.index(word)
                matrix[doc_idx][word_idx] += 1
    return matrix

def build_inverted_index(dictionary, incidence_matrix):
    inverted_index = {}
    for term_index, term in enumerate(dictionary):
        documents = []
        for doc_index, occurrence in enumerate(incidence_matrix[:, term_index]):
            if occurrence > 0:
                documents.append(doc_index + 1)  
        inverted_index[term] = documents
    return inverted_index

def boolean_search(query, inverted_index):
    import operator
    from functools import reduce

    def intersect(lists):
        return list(reduce(operator.and_, map(set, lists)))

    def union(lists):
        return list(reduce(operator.or_, map(set, lists)))

    tokens = query.lower().split()
    operators = {'and': intersect, 'or': union}
    current_op = None
    result = []

    for token in tokens:
        if token in operators:
            current_op = operators[token]
        elif token == 'not':
            next_token = next(tokens)
            result = [doc for doc in result if doc not in inverted_index.get(next_token, [])]
        else:
            if current_op:
                result = current_op([result, inverted_index.get(token, [])])
            else:
                result = inverted_index.get(token, [])

    return result



def complex_search(query, documents):
    # Création du modèle TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Transformer la requête en vecteur TF-IDF en utilisant le même vectorisateur
    query_vector = vectorizer.transform([query])

    # Calcul de la similarité cosinus entre la requête et les documents
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Récupération des indices des documents par ordre de pertinence
    relevant_docs_indices = cosine_similarities.argsort()[:-5:-1]  # Obtenir les indices des 4 documents les plus pertinents

    return relevant_docs_indices, cosine_similarities[relevant_docs_indices]



# Chemin vers les fichiers
path_to_files = './'  # Modifier selon l'emplacement réel des fichiers

# Lire et traiter chaque fichier
documents = []
for i in range(1, 11):  # Pour 1.txt à 10.txt
    file_path = os.path.join(path_to_files, f"{i}.txt")
    content = read_file(file_path)
    processed_tokens = preprocess_text(content)
    documents.append(processed_tokens)

# Créer un dictionnaire global à partir des tokens de tous les documents
dictionary = create_dictionary(documents)

# Construire la matrice d'incidence
incidence_matrix = build_incidence_matrix(documents, dictionary)
print(incidence_matrix)

inverted_index = build_inverted_index(dictionary, incidence_matrix)
print(inverted_index)




boolean_query = "covid or vaccination"
boolean_search_results = boolean_search(boolean_query, inverted_index)
print(boolean_search_results)


complex_query = "covid-19 strategies"
complex_search_results = complex_search(complex_query, inverted_index)
print(complex_search_results)
