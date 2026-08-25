import streamlit as st

from pyrag.wiring import build_pipeline

st.set_page_config(page_title="Python Docs RAG", page_icon="🐍")
st.title("🐍 Python Docs Q&A")


@st.cache_resource
def get_pipeline():
    pipeline, _ = build_pipeline()
    return pipeline


pipeline = get_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the Python standard library...")

if question and question.strip():
    with st.spinner("Retrieving and generating..."):
        result = pipeline.answer(question)
    st.session_state.history.append((question, result))

for past_question, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(past_question)

    with st.chat_message("assistant"):
        st.write(result.answer)

        if result.sources:
            with st.expander("Sources used"):
                for rc in result.sources:
                    st.markdown(
                        f"**{rc.chunk.section_title}** ({rc.chunk.source_file}) — score {rc.score:.2f}"
                    )
                    st.caption(rc.chunk.text[:300] + "...")
