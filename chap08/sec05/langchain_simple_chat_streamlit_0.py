import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,  # 기본 대화 기록 클래스
    InMemoryChatMessageHistory,  # 메모리에 대화 기록을 저장하는 클래스
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory  # 메시지 기록을 활용해 실행 가능한 wrapper 클래스
from langchain_openai import ChatOpenAI  # 오픈AI 모델을 사용하는 랭체인 챗봇 클래스

load_dotenv()

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자의 질문에 친절이 답하는 AI챗봇이다.")
    ]

# 세션별 대화 기록을 저장할 딕셔너리 대신 session_state 사용
if "store" not in st.session_state:
    st.session_state["store"] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = InMemoryChatMessageHistory()
    return st.session_state["store"][session_id]

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOpenAI(model=os.getenv("DEFAULT_MODEL"))
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config: RunnableConfig = {"configurable": {"session_id": "abc2"}}

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    print('user:', prompt)  
    st.session_state.messages.append(HumanMessage(prompt))
    st.chat_message("user").write(prompt)

    response = with_message_history.invoke([HumanMessage(prompt)], config=config)

    msg = response.content
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(msg)
    print('assistant:', msg)
