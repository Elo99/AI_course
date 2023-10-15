import numpy as np
from collections import Counter
import math

class TextDocumentAnalyzer:
    def __init__(self, document_paths):
        self.document_paths = document_paths
        self.documents = []
        self.vocabulary = []
        self.word_frequencies = None
        self.tf_matrix = None
        self.idf_values = None
        self.tfidf_matrix = None
        self.cosine_similarities = None
        self.jaccard_similarities = None
        self.cosine_min_distance = np.inf
        self.jaccard_min_distance = 1.0
        self.cosine_min_pair = (0, 0)
        self.jaccard_min_pair = (0, 0)
        self.cosine_max_distance = -1.0
        self.jaccard_max_distance = 0.0
        self.cosine_max_pair = (0, 0)
        self.jaccard_max_pair = (0, 0)

    def tokenize_documents(self):
        self.documents = []
        for path in self.document_paths:
            with open(path, 'r') as file:
                text = file.read()
                tokens = text.split()
                self.documents.append(tokens)

    def create_vocabulary(self):
        self.vocabulary = list(set(word for doc in self.documents for word in doc))

    def calculate_word_frequencies(self):
        N = len(self.documents)
        self.word_frequencies = np.zeros((N, len(self.vocabulary)))

        for i, doc in enumerate(self.documents):
            doc_word_count = Counter(doc)
            for j, word in enumerate(self.vocabulary):
                self.word_frequencies[i, j] = doc_word_count[word]

    def generate_tf_matrix(self):
        self.tf_matrix = self.word_frequencies / self.word_frequencies.sum(axis=1)[:, np.newaxis]

    def calculate_idf_values(self):
        N = len(self.documents)
        self.idf_values = np.array([math.log(N / (1 + np.count_nonzero(self.word_frequencies[:, j]))) for j in range(len(self.vocabulary))])

    def calculate_tfidf_matrix(self):
        self.tfidf_matrix = self.tf_matrix * self.idf_values

    def calculate_cosine_and_jaccard_similarities(self):
        N = len(self.documents)
        self.cosine_similarities = np.zeros((N, N))
        self.jaccard_similarities = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                self.cosine_similarities[i, j] = self.cosine_similarity(self.tfidf_matrix[i], self.tfidf_matrix[j])
                doc_set1 = set(self.documents[i])
                doc_set2 = set(self.documents[j])
                self.jaccard_similarities[i, j] = self.jaccard_similarity(doc_set1, doc_set2)

    @staticmethod
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return dot_product / (norm_vector1 * norm_vector2)

    @staticmethod
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def find_min_max_similarities(self):
        N = len(self.documents)
        for i in range(N):
            for j in range(i + 1, N):
                if self.cosine_similarities[i, j] < self.cosine_min_distance:
                    self.cosine_min_distance = self.cosine_similarities[i, j]
                    self.cosine_min_pair = (i, j)
                if self.jaccard_similarities[i, j] < self.jaccard_min_distance:
                    self.jaccard_min_distance = self.jaccard_similarities[i, j]
                    self.jaccard_min_pair = (i, j)
                if self.cosine_similarities[i, j] > self.cosine_max_distance:
                    self.cosine_max_distance = self.cosine_similarities[i, j]
                    self.cosine_max_pair = (i, j)
                if self.jaccard_similarities[i, j] > self.jaccard_max_distance:
                    self.jaccard_max_distance = self.jaccard_similarities[i, j]
                    self.jaccard_max_pair = (i, j)

    def print_results(self):
        print(f"Documents with greatest Cosine Similarity: {self.cosine_max_pair}, Similarity: {self.cosine_max_distance}")
        print(f"Documents with greatest Jaccard Similarity: {self.jaccard_max_pair}, Similarity: {self.jaccard_max_distance}")
        print(f"Documents with smallest Cosine Similarity: {self.cosine_min_pair}, Similarity: {self.cosine_min_distance}")
        print(f"Documents with smallest Jaccard Similarity: {self.jaccard_min_pair}, Similarity: {self.jaccard_min_distance}")

# Usage example:
document_paths = [f'texts/{i}.txt' for i in range(1999)]
analyzer = TextDocumentAnalyzer(document_paths)
analyzer.tokenize_documents()
analyzer.create_vocabulary()
analyzer.calculate_word_frequencies()
analyzer.generate_tf_matrix()
analyzer.calculate_idf_values()
analyzer.calculate_tfidf_matrix()
analyzer.calculate_cosine_and_jaccard_similarities()
analyzer.find_min_max_similarities()
analyzer.print_results()
