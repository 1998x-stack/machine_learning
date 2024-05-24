import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple

def load_data(directory: str) -> Tuple[List[str], List[str]]:
    """Load the dataset from the given directory.
    
    Args:
        directory (str): Directory to load data from.
    
    Returns:
        Tuple[List[str], List[str]]: List of document contents and list of document labels.
    """
    documents = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='latin1') as f:
                documents.append(f.read())
                labels.append(os.path.basename(root))
    return documents, labels

def preprocess_data(documents: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Preprocess the documents to obtain a document-word matrix.
    
    Args:
        documents (List[str]): List of document contents.
    
    Returns:
        np.ndarray: Document-word matrix.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(documents)
    return X.toarray(), vectorizer.get_feature_names_out()

# 加载数据
documents, labels = load_data('directory/to/data')

# 预处理数据
X, feature_names = preprocess_data(documents)
print("Data preprocessing complete.")

class LDA:
    """Latent Dirichlet Allocation using NumPy.
    
    Attributes:
        n_topics (int): Number of topics.
        alpha (float): Hyperparameter for document-topic distribution.
        beta (float): Hyperparameter for topic-word distribution.
        n_iter (int): Number of iterations for Gibbs sampling.
    """
    
    def __init__(self, n_topics: int, alpha: float, beta: float, n_iter: int):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the LDA model to the given document-word matrix and transform it to topic distributions.
        
        Args:
            X (np.ndarray): Document-word matrix.
        
        Returns:
            np.ndarray: Document-topic distribution matrix.
        """
        n_docs, n_words = X.shape
        self.n_docs = n_docs
        self.n_words = n_words
        
        # Initialize the topic assignments randomly
        doc_topic_counts = np.zeros((n_docs, self.n_topics)) + self.alpha
        topic_word_counts = np.zeros((self.n_topics, n_words)) + self.beta
        topic_counts = np.zeros(self.n_topics) + n_words * self.beta
        
        topic_assignments = []
        for d in range(n_docs):
            current_doc_topics = []
            for w in range(n_words):
                if X[d, w] > 0:
                    topics = np.random.randint(0, self.n_topics, X[d, w])
                    current_doc_topics.append(topics)
                    for t in topics:
                        doc_topic_counts[d, t] += 1
                        topic_word_counts[t, w] += 1
                        topic_counts[t] += 1
            topic_assignments.append(current_doc_topics)
        
        # Gibbs sampling
        for _ in range(self.n_iter):
            for d in range(n_docs):
                for w in range(n_words):
                    if X[d, w] > 0:
                        current_word_topics = topic_assignments[d][w]
                        for t in current_word_topics:
                            doc_topic_counts[d, t] -= 1
                            topic_word_counts[t, w] -= 1
                            topic_counts[t] -= 1
                        prob = (doc_topic_counts[d, :] * topic_word_counts[:, w]) / topic_counts
                        new_topics = np.random.multinomial(X[d, w], prob / prob.sum())
                        new_topic = new_topics.argmax()
                        topic_assignments[d][w] = [new_topic]
                        doc_topic_counts[d, new_topic] += 1
                        topic_word_counts[new_topic, w] += 1
                        topic_counts[new_topic] += 1
        
        self.doc_topic_dist = doc_topic_counts / doc_topic_counts.sum(axis=1, keepdims=True)
        self.topic_word_dist = topic_word_counts / topic_word_counts.sum(axis=1, keepdims=True)
        
        return self.doc_topic_dist

def visualize_topics(doc_topic_dist: np.ndarray, n_topics: int):
    """Visualize the document-topic distribution using PCA.
    
    Args:
        doc_topic_dist (np.ndarray): Document-topic distribution matrix.
        n_topics (int): Number of topics.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(doc_topic_dist)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('Document-Topic Distribution Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# 运行LDA模型
lda_model = LDA(n_topics=5, alpha=0.1, beta=0.1, n_iter=100)  # Reduced topics and iterations for quick execution
doc_topic_dist = lda_model.fit_transform(X)
print("LDA fitting complete.")

# 可视化主题分布
visualize_topics(doc_topic_dist, lda_model.n_topics)
print("Topic visualization complete.")
