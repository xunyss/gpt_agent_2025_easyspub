import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory  # 메모리에 대화 기록을 저장하는 클래스
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory  # 메시지 기록을 활용해 실행 가능한 래퍼wrapper 클래스
from langchain_openai import ChatOpenAI  # 오픈AI 모델을 사용하는 랭체인 챗봇 클래스

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatOpenAI(model=os.getenv("DEFAULT_MODEL"))

# 세션별 대화 기록을 저장할 딕셔너리
store = {}

# 세션 ID에 따라 대화 기록을 가져오는 함수
# NOTE: RunnableWithMessageHistory. invoke() -> _merge_configs() -> get_session_history()
def get_session_history(session_id: str):
    # 만약 해당 세션 ID가 store에 없으면, 새로 생성해 추가함
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()  # 메모리에 대화 기록을 저장하는 객체 생성
    return store[session_id]  # 해당 세션의 대화 기록을 반환

# 모델 실행 시 대화 기록을 함께 전달하는 래퍼 객체 생성
with_message_history = RunnableWithMessageHistory(model, get_session_history)

config: RunnableConfig = {
    "configurable": {"session_id": "abc2"}
} # 세션 ID를 설정하는 config 객체 생성

response = with_message_history.invoke(
    input=[HumanMessage(content="안녕? 난 송정헌이야.")],
    config=config,
)
print(response.content)

response = with_message_history.invoke(
    input=[HumanMessage(content="내 이름이 뭐지?")],
    config=config,
)
print(response.content)
