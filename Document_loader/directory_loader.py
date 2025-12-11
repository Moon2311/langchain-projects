from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Document_loader/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = list(loader.lazy_load()) # convert to list
print("Total PDF documents:", len(docs))
print(docs[0].page_content)

# for document in docs:x
#     print(document.metadata)
