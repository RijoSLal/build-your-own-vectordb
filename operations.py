import os 
import numpy as np
import logger_setup
import logging 
import h5py
import pyarrow as pa
import pyarrow.parquet as pq 
import pyarrow.dataset as ds
import ollama
from typeguard import typechecked

logger_setup.setup_config() # config setup

logger = logging.getLogger(__name__)


class Operation:

    """
    this class provides functionality to create a collection directory, initialize
    storage files (`HDF5` for embeddings and `parquet` for metadata), and manage schema

    Args:
        collection_name (str): collection directory

    Attributes:
        _collection_name (str): collection directory name
        _max_length (int): maximum allowed length of embeddings or text
        _h5py (str): path to the `HDF5` file storing embeddings
        _parquet (str): path to the `parquet` file storing metadata
        schema (pyarrow.Schema): schema for the metadata `parquet` file
    """

    def __init__(self, collection_name: str):
        
        self._collection_name = collection_name

        self._max_length = 2048
        
        #-------------collection_dir----------------

        if os.path.isdir(self._collection_name):
            logger.warning(f"collection `{self._collection_name}` already exist!")
        else:
            try:
                os.makedirs(self._collection_name, exist_ok = True)
                logger.info(f"collection `{self._collection_name}` created successfully.")
            except OSError as e:
                logger.error(f"unable to create collection : {e}")

        #-------------------------------------------

        self._h5py = os.path.join(self._collection_name, "space.h5")

        self._parquet = os.path.join(self._collection_name, "space.parquet")

        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("meta", pa.map_(pa.string(), pa.string()))
        ])

        #----------------------------------------------
        if not os.path.exists(self._h5py):
            with h5py.File(self._h5py, "w"):
                pass  # create empty file 

    @typechecked
    def create_embedding(self, document: str) -> np.ndarray:

        """
        this method helps to convert documents into embedding

        Args:
            document (str): the text contents for which the embedding is to be generated
        
        Returns:
            np.ndarray : embedding vector of the text content
            
        """

        response = ollama.embeddings(model="all-minilm:l6-v2", prompt = document)
        return np.array(response.embedding)
    

    def is_valid(self, embedding: np.ndarray, file: h5py._hl.files.File) -> bool:
        """
        this method validate the embedding is compatiable or not by checking embedding shape, dimensions
        
        Args:
            embedding (np.ndarray): embedding vector 
            file (h5py._hl.files.File): `HDF5` file object that stores embeddings 
        
        Returns:
            bool : embedding is compatiable or not

        """
       
        if embedding.ndim != 1 or embedding.shape[0] > self._max_length:
            return False
        
        if len(file.keys()) == 0:
            return True 
        
        first_key = next(iter(file.keys()))

        first_shape = file[first_key].shape

        return embedding.shape == first_shape
        
    @typechecked
    def insertion(self, id: str = None, embedding: np.ndarray = None, metadata: dict = dict()) -> None:
        """
            insert a new embedding, id and its metadata into the collection

            this method adds the embedding to the `HDF5` file and appends its corresponding
            metadata to the `parquet file` and checks are performed to ensure the `id` does not
            already exist and that the embedding is valid

            Args:
                id (str): unique identifier for the embedding, cannot be None
                embedding (np.ndarray): embedding vector to store, cannot be None
                metadata (dict): optional metadata associated with the embedding

            Raises:
                TypeError: If `id` or `embedding` is None
        
            Returns:
                None
        """

        if id is None or embedding is None:
           raise TypeError("`id` or `embedding` cannot be empty!")
    
        try:
            with h5py.File(self._h5py, 'a') as file:
                
                if id in file.keys():
                    logger.warning("`id` already exist, skipping insertion...")
                    return None 
                
                elif not self.is_valid(embedding, file):
                    logger.warning('`embedding` is not compatible, skipping insertion...')
                    return None
                
                else:
                    file.create_dataset(id , data = embedding)

            if os.path.exists(self._parquet):
                table = pq.read_table(self._parquet)
                package = table.to_pylist()
            else:
                package = list()       

            package.append({
                "id": id, 
                "meta": metadata
            })

            pq.write_table(pa.Table.from_pylist(package, schema=self.schema), self._parquet)

            logger.info("insertion successful")

            return None
            
        except Exception as e:

            logger.error(f"insertion failed : {e}")
            
            return None
                 
    
            
    @typechecked
    def deletion(self, id: str = None) -> None:
        """
            delete an existing embedding, id and its metadata from the collection

            this method removes the embedding from the `HDF5` file and its corresponding
            metadata entry from the `parquet` file. The operation is skipped if the `id`
            does not exist.

            Args:
                id (str): unique identifier of the embedding to delete that cannot be `None`

            Raises:
                TypeError: if `id` is `None`

            Returns:
                None
        """
        if id is None:
           raise TypeError("`id` cannot be empty!")
        
        try:
            with h5py.File(self._h5py, 'r+') as file:
                
                if id not in file.keys():
                    logger.warning("`id` does not exist, abort deletion!")
                    return None
                else:
                    del file[id]

                    table = pq.read_table(self._parquet)
                    rows = [r for r in table.to_pylist() if r["id"] != id]
                    pq.write_table(pa.Table.from_pylist(rows, schema=self.schema), self._parquet)
                    
                    logger.info("deletion successful...")
                            

        except Exception as e:
            logger.error(f"deletion failed : {e}")
            return None
        
    @typechecked
    def updation(self, id: str = None, embedding: np.ndarray = None, metadata: dict = dict()) -> None:
        """
            update an existing embedding and its metadata in the collection.

            this method replaces the existing embedding vector in the `HDF5` file and updates
            its corresponding metadata entry in the `parquet` file. The operation is skipped
            if the specified `id` does not exist or if the new embedding is invalid.

            Args:
                id (str): unique identifier of the embedding to update,cannot be None.
                embedding (np.ndarray): new embedding vector to overwrite the existing one.
                metadata (dict): updated metadata associated with the embedding.

            Raises:
                TypeError: If `id` or `embedding` is None.

            Returns:
                None
        """

        if id is None or embedding is None:
           raise TypeError("`id` or `embedding` cannot be empty!")
        
        try:
            with h5py.File(self._h5py, 'r+') as file:
                
                if id not in file.keys():
                    logger.warning("`id` does not exist, skipping updation...")
                    return None
                
                if not self.is_valid(embedding, file):
                    logger.warning('`embedding` is not compatible, skipping updation...') 
                    return None
                
                else:
                    file[id][:] = embedding

                table = pq.read_table(self._parquet)
                rows = table.to_pylist()

                for r in rows:
                    if r["id"] == id:
                        r["meta"] = metadata

                table = pa.Table.from_pylist(rows, schema=self.schema)
                pq.write_table(table, self._parquet)
                logger.info("updation successful...")

        except Exception as e:
            logger.error(f"updation failed : {e}")
            return None


        
    def filter(self, id: str, metadata: str) -> np.ndarray:
        pass 