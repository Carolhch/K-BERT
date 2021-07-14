from Bert_embedding import BertEmbedding
from word_embedding_skip_gram import Word2Vec
from word_vectors_Chinese_Word_Vectors import ChineseWordVector

class EmbeddingFactory():
	def __init__(self,embedding_type,folder_name):
		self.embedding_type = self.init(embedding_type,embedding_name)

	def init(self,type, name):
		path = self.get_model_path(name)
		if type == "Bert":
            self.embedding = BertEmbedding(model_name=path)
        elif embedding_type == "ChineseWordVector":
            self.embedding = ChineseWordVector('word_vectors/merge_sgns_bigram_char300.txt', 0)
        else:
            self.embedding = Word2Vector(spo_files)

    def get_model_path(self,model_name):
    	this_dir = dirname(__file__)
		return os.path.join(this_dir,'..',"models",model_name)