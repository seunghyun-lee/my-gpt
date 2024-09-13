import streamlit as st

def hide_sidebar():
    st.set_page_config(initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {display: none}
        #MainMenu {visibility: hidden;}
        .css-zt5igj {display: none;}
        </style>
    """,
        unsafe_allow_html=True,
    )

def show_sidebar():
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {display: flex}
        #MainMenu {visibility: visible;}
        .css-zt5igj {display: block;}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [x] [QuizGPT](/QuizGPT)
- [x] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [x] [InvestorGPT](/InvestorGPT)
"""
)