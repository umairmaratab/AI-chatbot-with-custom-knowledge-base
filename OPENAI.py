from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 300
    max_chunk_overlap = 20
    chunk_size_limit = 600

    #define LLM

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens = num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit = chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTVectorStoreIndex.load_from_disk('index.json')
    while True:
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode = "compact")
        display(Markdown(f"Response: <b> {response.response}</br>"))



os.environ["OPENAI_API_KEY"] = input("Paste your key here and press enter:")
construct_index("Questions")

ask_ai()