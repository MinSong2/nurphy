import treform as ptm
from nurphy.word_embeddings.node2vecModel import Node2VecModel

embedding_filename='./node2vec.emb'
n2vec = Node2VecModel()
n2vec.load_model(embedding_filename)
results= n2vec.most_similars('트럼프')
print(results)

pair_similarity = n2vec.compute_similarity('트럼프', '무섭다')
for pair in pair_similarity:
    print(str(pair[0]) + " -- " + str(pair[1]))