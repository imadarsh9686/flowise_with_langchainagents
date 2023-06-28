import streamlit as st


# Create a function for each page
def home():
    
    import os
    from langchain.text_splitter import CharacterTextSplitter

    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import TextLoader
    from langchain.vectorstores import Pinecone
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import ConversationChain
    from langchain.chains.conversation.memory import ConversationBufferMemory
    from langchain.document_loaders import YoutubeLoader

    from langchain.chains.summarize import load_summarize_chain
    from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
    import pinecone
    #import nltk

    #nltk.download('punkt')
    # openai_api_key = "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"

    os.environ["PINECONE_API_KEY"] = pineconekey
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.title("🦜")
    # Add content specific to the home page
    # import streamlit as st
    import requests

    API_URLS = [
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/8f17b231-6b0c-4ab6-929d-214d368e111e",
        # DOCgpt
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/6a421494-72c9-42f1-9520-a84591bbdc54",
        # GoogleGPT
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/8f17b231-6b0c-4ab6-929d-214d368e111e"
        # finetune answer from doc - web - openai

    ]

    # Initialize the selected API index
    selected_api_index = 2

    def query(payload):

        try:
            response = requests.post(API_URLS[0], json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as err:
            st.error("Error occurred:", err)
            return None

    def query1(payload):
        try:
            response = requests.post(API_URLS[1], json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as err:
            st.error("Error occurred:", err)
            return None

    # st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # List to hold the conversation history
    conversation = []
    chatlist = []

    if selected_api_index == 2:
        # Chat container

        # PDF UPLOADER

        docs_chunks = []

        # openai_api_key = os.environ.get('API_KEY')
        # openai_api_key = "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"
        # pineconekey = "f4e3f5b8-fc9a-4d6d-be18-ba5f200e0e52"
        # pineconeEnv = "us-west1-gcp-free"

        # Initialize Pinecone
        pinecone.init(api_key=pineconekey, environment=pineconeEnv)
        # index_name2 = "babyagi"

        # embeddings

        embeddings = OpenAIEmbeddings()

        # image = Image.open('ai.png')
        # st.image(image, caption='AI', width=200)

        with st.sidebar:
            docs_chunks1 = []
            loader_choice = st.radio("Select type of link🔗", ["🛑-Youtube URL", "🌐-Blog/website URL"])

            

            url= st.text_input("🔗 Paste URL here:")

            if loader_choice == "🛑-Youtube URL":
                if st.button("(RUN)✅ Press Here to Run"):
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    data = loader.load()
                    print("Data loaded successfully with youtubeloader.")

                    text_splitter = CharacterTextSplitter(separator='\n',
                                                          chunk_size=1000,
                                                          chunk_overlap=100)

                    docs = text_splitter.split_documents(data)
                    docs_chunks1.extend(docs)
                    #st.write(docs)
                    #pinecone.init(api_key=pineconekey, environment=pineconeEnv)

                    index3 = Pinecone.from_documents(docs_chunks1, embeddings, index_name=index_name2)

                    docs_chunks1.clear()


            if loader_choice=="🌐-Blog/website URL":
                


                urls_list = st.session_state.get('urls_list', [])

                if st.button("(ADD)🔼 Press Here to add url to the list "):
                    urls_list.append(url)
                    st.session_state['urls_list'] = urls_list
                    st.write(urls_list)

                if st.button("(CLEAR)🧹 Clear the list"):
                    st.session_state['urls_list'] = []
                    st.write("List is empty now. Please paste your URLs one by one.")

                if st.button("(RUN)✅ Press Here to Run"):
                    urls = [

                        'https://cobusgreyling.medium.com/openai-function-calling-98fbf9539d2a'
                    ]

                    try:
                        loaders = UnstructuredURLLoader(urls=urls)
                        data = loaders.load()
                        print("Data loaded successfully with UnstructuredURLLoader.")
                        st.write(data)

                    except Exception as e:
                        st.write("URL not supported")
                        print("Error loading data with UnstructuredURLLoader:", e)
                        print("Trying with SeleniumURLLoader...")

                        try:
                            loader = SeleniumURLLoader(urls=urls)
                            data = loader.load()
                            print("Data loaded successfully with SeleniumURLLoader.")
                        except Exception as e:
                            st.write("URL not supported")
                            print("Error loading data with SeleniumURLLoader:", e)
                            print("Both loaders failed to load data.")

                    text_splitter = CharacterTextSplitter(separator='\n',
                                                          chunk_size=1000,
                                                          chunk_overlap=100)
                    docs = text_splitter.split_documents(data)

                    docs_chunks1.extend(docs)
                    

                    index4 = Pinecone.from_documents(docs_chunks1, embeddings, index_name=index_name2)


                    docs_chunks1.clear()





        def process_file(uploaded_file):

            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            _, file_extension = os.path.splitext(uploaded_file.name)

            # write the uploaded file to disk
            with open(uploaded_file.name, 'wb') as f:
                f.write(bytes_data)

            documents = None
            if file_extension.lower() == '.pdf':
                # Load the PDF file with PyPDF Loader
                loader = PyPDFLoader(uploaded_file.name)
                documents = loader.load()

            elif file_extension.lower() == '.txt':
                # Load the text file with TextLoader
                loader = TextLoader(uploaded_file.name, encoding='utf8')
                documents = loader.load()

            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            return documents

        def split_docs(documents, chunk_size=1000, chunk_overlap=100):
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            docs = text_splitter.split_documents(documents)
            return docs

        uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
        if st.button('upload'):

            if len(uploaded_files) > 0:
                for uploaded_file in uploaded_files:
                    documents = process_file(uploaded_file)
                    docs_chunks.extend(split_docs(documents))
            index1 = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name2, overwrite=True)




    # selected_api_index == 4

    # User input
    user_input = st.text_input("User:", key="user_input")
    submit_button = st.button("ASK", key="submit_button")

    # if selected_api_index==4:

    if submit_button and user_input:
        response = query({"history": conversation, "question": user_input})
        response2 = "Hello! How can I assist you today?"
        response1 = query1({"history": conversation, "question": user_input})
        # response1 = query1({"history": conversation, "question": user_input})
        # Add user input to conversation history
        conversation.append({"role": "user", "question": user_input})
        # Query the selected API

        if selected_api_index == 2:

            if response or response1 is not None:
                # Add bot response to conversation history
                chatlist.append({"Dtllm_result": response2, "document result": response})
            else:
                chatlist.append(
                    {
                        "result": "bot",
                        "content": "Sorry, I am unable to process your request at the moment.",
                    }
                )

            model_name = "gpt-3.5-turbo"
            llm = ChatOpenAI(temperature=0.2, model_name=model_name)
            conversation1 = ConversationChain(
                llm=llm,
                verbose=True,
                #memory=ConversationBufferMemory()
            )

            No="No"

            answer1 = conversation1.predict(
                input=f"For the question{user_input} just answer it from this content given only give the content answer  {chatlist} if u dont feel answer is correct or do not found the answer in the context provided or if the quetion is realted to realtime information strictly give strictly output with word No ")
                 #input=f"your goal is to provide accurate responses to user queries. you will utilize the provided content, specifically the {chatlist}, to generate an answer based on the user's {user_input}. If you can find a suitable answer in the content, you will provide it as the output. However, if you cannot find a relevant answer or struggle to provide an appropriate response, you will strictly give output with word {No} Please note that your responses will be limited to either an accurate answer from the content or a straightforward word {No} if the answer cannot be determined.")
                #input=f" you are an chat bot your work is to give correct response to the user from this  user input :- {user_input} just answer it from this content given output answer should be either answer from content or give No as output if you donot know the answer  this the content:-{chatlist} if u dont feel answer/response is correct for the user input or do not find the answer in the content strictly give strictly output with word No donot make any sentences")
            st.write("answer 1", answer1)
            chatlist.clear()

            



            if answer1=="No." :
                if response1 or response is not None:
                    # Add bot response to conversation history
                    chatlist.append({"internet_result": response1, "document result": response})
                else:
                    chatlist.append(
                        {
                            "result": "bot",
                            "content": "Sorry, I am unable to process your request at the moment.",
                        }
                    )

                model_name = "gpt-3.5-turbo"
                llm = ChatOpenAI(temperature=0.2, model_name=model_name)
                conversation1 = ConversationChain(
                    llm=llm,
                    verbose=True,
                    #memory=ConversationBufferMemory()
                )

                answer2 = conversation1.predict(
                    input=f"For the question{user_input} just answer it from this content given {chatlist} strictly give output between internet_result and document result just give content answer")
                st.write("answer 2 ", answer2)


                chatlist.clear()

            






# Create a dictionary mapping page names to the corresponding functions
pages = {
    "🦜": home
}
st.sidebar.title("Paste your URL 🤖 BELOW")
# Add a sidebar to navigate between pages
page = st.sidebar.radio(".", options=list(pages.keys()))

with st.sidebar:
    openai_api_key = st.secrets["openai_api_key"]
    pineconekey = st.secrets["pineconekey"]
    pineconeEnv = "us-west1-gcp-free"
    index_name2 = "axstream"
    serp_api = st.secrets["serp_api"]

if openai_api_key and pineconekey and pineconeEnv and index_name2 and serp_api:

    st.success("!")
    pages[page]()

    # You can use the API keys in your code here
    # For example, make API requests using the keys

    # ...
else:
    st.warning("Please fill correct API keys .")

# Execute the function corresponding to the selected page
# pages[page]()
