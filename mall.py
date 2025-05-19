import streamlit as st
import wikipedia
import re
from pydantic import BaseModel
from typing import List, Optional
import cohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings

st.set_page_config(
        page_title=" rajesh's Chat Bot",
        page_icon="🔴",
        layout="centered"
    )

# === Dropdown Menu for App Selection ===
app_selection = st.selectbox(
    "Select Application",
    ("Institution Info Finder", "Rotaract Chat Bot")
)

# === Institution Info Finder App ===
if app_selection == "Institution Info Finder":
    

    # 1: Define Pydantic 
    class InstitutionInfo(BaseModel):
        institution_name: str
        founder: str
        founded_year: Optional[int]
        branches: List[str]
        employee_count: Optional[int]
        summary: str

    # 2: Extract Info from Wikipedia
    def get_institution_info(name: str) -> Optional[InstitutionInfo]:
        try:
            page = wikipedia.page(name)
            summary = wikipedia.summary(name, sentences=4)
            content = page.content
        except:
            return None

        
        def extract_founder():
            match = re.search(r'founded by ([\w\s\.]+)', content, re.IGNORECASE)
            return match.group(1).strip() if match else "Unknown"

        def extract_founded_year():
            match = re.search(r'founded in (\d{4})', content, re.IGNORECASE)
            return int(match.group(1)) if match else None

        def extract_branches():
            matches = re.findall(r'\b(?:Campus|Branch)\s+([A-Za-z\s]+)', content)
            return list(set(matches))  # Remove duplicates

        def extract_employees():
            match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s+(?:employees|staff|workers)', content, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(',', '')
                return int(num_str)
            return None

        
        return InstitutionInfo(
            institution_name=name,
            founder=extract_founder(),
            founded_year=extract_founded_year(),
            branches=extract_branches(),
            employee_count=extract_employees(),
            summary=summary
        )

    # === Streamlit App ===
   
    st.title("Institution Information Finder")
    st.write("Enter the name of an institution to fetch details from Wikipedia.")

    institution_name = st.text_input("Institution Name")

    if st.button("Fetch Info"):
        if institution_name.strip() == "":
            st.warning("Please enter an institution name.")
        else:
            with st.spinner("Fetching data from Wikipedia..."):
                info = get_institution_info(institution_name)

            if info:
                st.write(info.summary)

                st.subheader("🔍 Details")
                st.json(info.model_dump())
            else:
                st.error("❌ Institution not found or data could not be retrieved.")


# === Rotaract Chat Bot App ===
elif app_selection == "Rotaract Chat Bot":
       

    COHERE_API_KEY = "4sdfokOEwmY7Z3gC3AtNa6TUsuLtx9Q6IbaqZB1X"
    co = cohere.Client(COHERE_API_KEY)

    class CustomCohereEmbeddings(Embeddings):
        def embed_documents(self, texts):
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    embedding = CustomCohereEmbeddings()

    llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

    st.title("Ask me about rotaract")

    file_path = "rotaract.txt"

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            file_content = file.read()

        loader = TextLoader(file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embedding)
        retriever = vectorstore.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        for q, a in st.session_state.history:
            st.write(f"**You:** {q}")
            st.write(f"**AI:** {a}")

        user_input = st.text_input("Ask something:")

        if user_input:
            answer = qa.run(user_input)
            st.session_state.history.append((user_input, answer))
            st.session_state.user_input = ""
            st.write(f"**You:** {user_input}")
            st.write(f"**AI:** {answer}")

    except FileNotFoundError:
        st.write(f"File {file_path} not found. Please place the file in the correct location.")