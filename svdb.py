import numpy as np
import logging
import logger_setup
import operations
import h5py
import pyarrow.dataset as ds
import pyarrow.parquet as pq 
from typeguard import typechecked

logger_setup.setup_config()

logger = logging.getLogger(__name__)

class Collection(operations.Operation):

    """
    represents a vector collection that supports iteration over stored items.

    inherits from `operations.Operation` and provides an iterator interface
    to access all entries (id, embedding, metadata and other attr) stored in
    the collection.

    Args:
        collection_name (str): collection directory name

    Attributes:
        _collection_name (str): collection directory inherited from the base class
    """

    def __init__(self, collection_name):
        super().__init__(collection_name)
        self._collection_name = collection_name

    def __iter__(self):
        """
        iterate over all entries in the collection to view each stored item

        Yields:
            str: stringified dictionary containing:
                - `id` (str): unique identifier for the embedding
                - `embedding` (tuple): shape of the embedding vector
                - `meta` (dict): metadata associated with the embedding
        """
        with h5py.File(self._h5py, "r") as file:
            
            table = pq.read_table(self._parquet)
            for row in table.to_pylist():
                 yield str({
                    "id": row["id"],
                    "embedding": file[row["id"]].shape,
                    "meta": dict(row["meta"])
                })
    
    @typechecked
    def top_k(self, embedding: np.ndarray, k: int = 3) -> list:
        
        """
            retrieve the top-k most similar entries from the collection

            this method compares the provided embedding against all stored embeddings
            using the defined similarity function and returns the top-k entries with the
            highest similarity scores, along with their corresponding metadata.

            Args:
                embedding (np.ndarray): the query embedding used for similarity comparison.
                k (int, optional): the number of top similar entries to return. Defaults to 3.

            Returns:
                list[dict]: list of dictionaries, each containing:
                    - `id` (str): the unique identifier of the entry.
                    - `similarity_score` (float): the similarity value between the query and stored embedding.
                    - `metadata` (dict): associated metadata for the entry.
        """
        
        with h5py.File(self._h5py, "r") as file:
            if not self.is_valid(embedding, file):
                logger.warning("`embedding` is not compatible, aborting top-k search!")
                return None
            if len(file.keys()) < k:
                logger.error(f"insufficient entries, expected at least {k} entries, but got {len(file.keys())}")
                return None
            
            results = [(key, self.similarity_function(value[:], embedding)) for key, value in file.items()]
           

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        dataset = ds.dataset(self._parquet, format="parquet")

        top_k_list = [
            {
                "id" : id,
                "similarity_score" : sim,
                "metadata" : dict(dataset.to_table(filter=ds.field("id")==id)["meta"][0].as_py())
            }
            for id, sim in sorted_results
        ] 

        return top_k_list


    @staticmethod
    @typechecked
    def similarity_function(embedding_x: np.ndarray, embedding_y: np.ndarray, similarity_function:str = 'cosine') -> np.float64:
        """
            compute the similarity or distance between two embedding vectors.

            this method supports multiple similarity or distance metrics, including
            cosine similarity, euclidean distance, manhattan distance, and dot product
            which returns a scalar value representing how similar or distant
            the two embeddings are

            Args:
                embedding_x (np.ndarray): the first embedding vector
                embedding_y (np.ndarray): the second embedding vector
                similarity_function (str, optional): the metric to use for comparison
                    supported values are:
                        - `"cosine"`: Cosine similarity (default)
                        - `"euclidean"`: Euclidean distance
                        - `"manhattan"`: Manhattan distance
                        - `"dot"`: Dot product

            Raises:
                ValueError: if an unsupported similarity function name is provided

            Returns:
                np.float64: numeric value representing the similarity
                
        """
        match similarity_function:

            case "cosine":

                # cosine_similarity = (A · B) / (||A|| * ||B||)

                return np.dot(embedding_x, embedding_y) / (np.linalg.norm(embedding_x) * np.linalg.norm(embedding_y))
              
            case "euclidean":
                
                # euclidean_distance = sqrt(Σ (A_i - B_i)²)

                return np.sqrt(np.sum((embedding_x - embedding_y)**2))

            case "manhattan":

                # manhattan_distance = Σ |A_i - B_i|

                return np.sum(np.abs(embedding_x - embedding_y))

            case "dot":

                #dot_product = Σ (A_i × B_i)

                return np.dot(embedding_x, embedding_y)

            case _:
                raise ValueError("similarity function doesn't exist!")
    

#--------------dummy-test-----------------------------------

# data = [
#     {"id": "1", "meta": {"source": "api", "version": "1"}, "array": [1, 2, 3, 4, 5]},
#     {"id": "2", "meta": {"source": "ui", "version": "2"}, "array": [5, 4, 3, 2, 1]},
#     {"id": "3", "meta": {"source": "batch", "version": "1"}, "array": [9, 8, 7, 6, 5]},
#     {"id": "4", "meta": {"source": "mobile", "version": "3"}, "array": [0, 1, 0, 1, 0]},
#     {"id": "5", "meta": {"source": "api", "version": "2"}, "array": [2, 4, 6, 8, 10]},
#     {"id": "6", "meta": {"source": "cron", "version": "1"}, "array": [3, 3, 3, 3, 3]},
#     {"id": "7", "meta": {"source": "etl", "version": "5"}, "array": [7, 6, 5, 4, 3]},
#     {"id": "8", "meta": {"source": "webhook", "version": "1"}, "array": [1, 0, 1, 0, 1]},
#     {"id": "9", "meta": {"source": "import", "version": "4"}, "array": [4, 8, 12, 16, 20]},
#     {"id": "10", "meta": {"source": "cli", "version": "2"}, "array": [2, 3, 5, 7, 11]}
# ]


# emb = np.array([3, 5, 7, 10, 15])

# col = Collection("hello")

# for i in data:
#     if i["id"]!="10":
#        col.insertion(i["id"],np.array(i["array"]),i["meta"])
#     else:
#         col.insertion(i["id"],np.array(i["array"]))

# print("inserted everything with 10 no metadata")
# table = pq.read_table(col._parquet)
# ids = [v.as_py() for v in table.column("id")]
# print(ids)


# metas = [v.as_py() for v in table.column("meta")]
# print(metas)

# #----------------------------------------------------------
# new =  np.array([1, 2, 3, 4, 5])
# new_meta = {"source": "mobiled", "version": "3s"}

# col.updation("4",new,new_meta)
# table = pq.read_table(col._parquet)


# print("updated 4 which was 'meta': {'source': 'mobile', 'version': '3'}, 'array': [0, 1, 0, 1, 0] to [1, 2, 3, 4, 5] and {'source': 'mobiled', 'version': '3s'} ")

# ids = [v.as_py() for v in table.column("id")]
# print(ids)


# metas = [v.as_py() for v in table.column("meta")]
# print(metas)

# #------------------------------------------------------------
# col.deletion("7")

# table = pq.read_table(col._parquet)

# print("deleted 7")
# ids = [v.as_py() for v in table.column("id")]
# print(ids)


# metas = [v.as_py() for v in table.column("meta")]
# print(metas)

# #-------------------------------------------------------------
# print(col.top_k(emb))



# ids = [v.as_py() for v in table.column("id")]
# print(ids)
# print("")

# metas = [v.as_py() for v in table.column("meta")]
# print(metas)
# print("")

# intel = iter(col)
# print(next(intel))
# print(next(intel))
