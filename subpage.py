import streamlit as st
from streamlit import session_state as state

# Define page functions
def page1():
    st.title('Page 1')
    st.write('This is the first page.')

def page2():
    st.title('Page 2')
    st.write('This is the second page.')

# Define the app
def main():
    st.sidebar.title('Navigation')
    pages = {
        'Page 1': page1,
        'Page 2': page2,
    }

    # Initialize state
    if 'page' not in state:
        state.page = 'Page 1'

    # Render the selected page
    pages[state.page]()

    # Create a menu with page selection
    page_options = list(pages.keys())
    new_page = st.sidebar.selectbox('Select Page', page_options)
    if new_page != state.page:
        state.page = new_page

if __name__ == '__main__':
    main()
