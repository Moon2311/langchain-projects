from langchain.text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dlangchain_text_spliter/l-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator='\n'
)

result = splitter.split_documents(docs)

print(result[1].page_content)