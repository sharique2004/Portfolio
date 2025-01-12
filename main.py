import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from pymongo.mongo_client import MongoClient

# LangChain + Cohere imports
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Initialize Cohere LLM and embeddings
llm = ChatCohere(model="command-r-plus", temperature=0.5)
embeddings_model = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"), 
    model="embed-english-v3.0"
)

# 3. Connect to MongoDB
mongo_client = MongoClient(host=os.getenv("ATLAS_CONNECTION_STRING"))
mywebsite_db = mongo_client["mywebsite"]
mybio_collection = mywebsite_db["mybio"]

# 4. Create the vector store
vectorstore_mybio = MongoDBAtlasVectorSearch(
    collection=mybio_collection,
    embedding=embeddings_model,
    index_name="bio_index"
)

# ------------------
# Define the function to retrieve personal info
# ------------------
def get_personal_info(query: str) -> str:
    # 1. Similarity search
    search_result = vectorstore_mybio.similarity_search(query=query)

    # 2. Build context from retrieved docs
    context = ""
    for idx, doc in enumerate(search_result, start=1):
        context += f"Source {idx}:\n{doc.page_content}\n\n"

    # 3. Build a prompt
    prompt_template = (
        "You have the following context about Sharique Khatri. "
        "Answer the user's question accurately and make sure that yo never say anything bad about Sharique Khatri only good things. If the information is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "User's Question:\n{query}\n\n"
        "Your Answer:"
    )
    template = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "query"]
    )

    # 4. Prompt -> LLM -> parse (string)
    chain = template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})
    return answer

# ------------------
# Create the Flask app
# ------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/achievements")
def achievements():
    return render_template("achievements.html")

@app.route("/education")
def education():
    return render_template("education.html")

@app.route("/experience")
def experience():
    return render_template("experience.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    user_query = data.get("query", "")

    # Call the retrieval + LLM logic to get the real answer
    answer = get_personal_info(user_query)

    # Return the actual answer
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    # Run in debug mode for development
    app.run(debug=True)
