import genism
from genism.models.doc2vec import Doc2Vec , TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

f= open('dataset.txt','r')
print(f.read)

corpus = [
    "This is first Sentence",
    "This is second Sentence",
    "This is third Sentence",
    "This is fourth Sentence",
    "This is fifth Sentence",
]

documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(corpus)]
model = Doc2Vec(documents , vector_size = 10 , window = 2 , min_count = 1 , workers =4)
model.save('sentenceEmbedderModel.pkl')
print('Model Creation Successful.' + repr(model))

vector = model.infer_vector(['this is not a sentence'])
vector_2 = model.infer_vector(['this is not a first sentence'])
vector_3 = model.infer_vector(['this is not a sentence'])


print("vector is " + repr(vector))
print("1 vs 2 " + repr(cosine_similarity([vector],[vector_2])))
print("1 vs 3 " + repr(cosine_similarity([vector],[vector_3])))