# Build-Your-Own-VectorDB 🗃️

**Build-Your-Own-VectorDB** is a fun weekend project created in just 2 days to challenge myself to build a vector database from scratch. The main goal was to experiment with some interesting ideas around **embedding-based automatic knowledge graph creation**, tackling fundamental problems faced in RAG (Retrieval-Augmented Generation).  

I didn’t think I would be able to spin up a vector database from scratch within a day and my crazy idea would solve the fundamental problems in RAG, but here we are! 💪

### Features ✨
- Store embeddings efficiently in **H5Py** files for high performance.  
- Save metadata in **Parquet** format for scalability and easy querying.  
- Simple features **insert, update, delete, and filter**.  
- Support for **top-k similarity search** using multiple similarity functions and default embedding model.  
- Lays the groundwork for **automatic knowledge graph construction** from embeddings.  

### Limitations ⚠️
- Many areas for improvement remain.  
- Feature set is minimal and primarily for experimentation.  
- Designed as a proof-of-concept, not production-ready.  

### Contributing 🤝
Feel free to contribute! Bug fixes, optimizations, or new features are all welcome. Let’s make this little weekend experiment even better.  

### Why `H5Py` &  `Parquet` 🛠️
- Embeddings are stored in `H5Py` files and metadata is stored in `Parquet` files which makes it scalable, easy to integrate with your RAG pipeline and for experimenting with semantic knowledge graphs.  

---

**Made with ❤️ over a weekend, with plenty of room to grow!**
