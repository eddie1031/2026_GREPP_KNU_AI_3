from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents.base import Document


SYSTEM_PROMPT = """당신은 쇼핑몰 고객센터 AI 어시스턴트입니다.
고객의 질문에 정확하고 친절하게 답변하세요.

규칙:
1. 반드시 아래 제공된 [참고 문서]의 내용을 바탕으로 답변하세요.
2. 참고 문서에 없는 내용은 "죄송합니다, 해당 내용은 확인이 어렵습니다. 고객센터(1588-0000)로 문의해 주세요."라고 안내하세요.
3. 답변은 명확하고 간결하게 작성하세요.
4. 고객이 불편함을 겪고 있다면 먼저 공감을 표현하세요.
5. 필요시 단계별(1, 2, 3...)로 안내하세요.

[참고 문서]
{context}"""

def build_rag_chain(vectorstore):
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.1-8b-instant"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 3
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])


    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def format_docs(docs: list[Document]) -> str:

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    sections = []

    for i, doc in enumerate(docs, 1):
        category = doc.metadata.get("category", "일반")
        question = doc.metadata.get("question", "")
        section = f"{i}. 카테고리: {category}"
        if question:
            section += f"  - 관련 질문: {question}"
        section += f"\n{doc.page_content}"
        sections.append(section)

    return "\n\n".join(sections)