from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings

from typing import List


def load_documents() -> List[Document]:
    import json

    loader = TextLoader('libs/004/qna_data.json')
    loaded_doc = loader.load()    

    json_docs = json.loads(loaded_doc[0].page_content)

    return [

        Document(
            page_content=f'질문: {doc['question']}\n\n 답변: {doc['answer']}',
            metadata= {
                'id': doc['id'],
                'category': doc['category'],
                'keywords': doc['keywords']
            }
        )
        for doc in json_docs

    ]


def split_docs(docs: List[Document]) -> List[Document]:
    """
    만일 문서가 충분히 크다면 청킹을 위해서 사용할 수 있습니다.
    """
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 50
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    return splitter.split_documents(docs)


def embedding(docs: List[Document]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore


def save_vector_to_local(vectorstore):
    path_str = './exp-faiss'    
    vectorstore.save_local(path_str)


def load_vector_from_local():
    path_str = './exp-faiss'
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def init_vectorstore():
    docs = load_documents()
    # split_documents = split_docs(docs)
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore

