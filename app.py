from vectorstore import init_vectorstore, load_vector_from_local
from chain import build_rag_chain

# init_vectorstore()

vectorstore = load_vector_from_local()
chain = build_rag_chain(vectorstore=vectorstore)

q1 = "제가 제주도에 살고있는데, 배송이 얼마나 걸릴까요?"
result = chain.invoke(q1)

print(result)

q2 = "물건이 맘에 안들어서 법적조치를 하고 싶어요. 방법이 뭔가요?"
result = chain.invoke(q2)

print(result)

"""
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
  from pydantic.v1.fields import FieldInfo as FieldInfoV1
전체 실행 결과:
제주도에 살으시네요. 저희 쇼핑몰의 일반 배송은 주문 후 2-3일 내 배송됩니다. 제주도 및 도서산간 지역은 추가 1-2일이 소요될 수 있습니다. 

예를 들어, 주문하신 상품이 1일 늦은 밤에 도착한 경우에도 1일 내에 제주도로 배송될 것입니다. 새벽 배송 서비스를 이용하시면 당일 오전 7시 이전에 받아보실 수 있습니다. 

배송 상태는 마이페이지의 주문조회에서 실시간으로 확인 가능합니다.
죄송합니다, 해당 내용은 확인이 어렵습니다. 고객센터(1588-0000)로 문의해 주세요.
"""