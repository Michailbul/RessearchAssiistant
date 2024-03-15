
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import FastAPI
import uvicorn
from langchain.retrievers import ArxivRetriever

from langserve import add_routes
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import json
import os

from prompts import SUMMARY_TEMPLATE, SUMMARY_TEMPLATE, SEARCH_PROMPT, WRITER_SYSTEM_PROMPT, RESEARCH_REPORT_TEMPLATE, SUMMARY_TEMPLATE_ARXIV, SUMMARY_PROMPT_ARXIV
from prompts import final_prompt as prompt 
from utils import web_search, scrape_text, collapse_list_of_lists


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = "ResearchAssistant"

#----------------------Arxiv
retriever = ArxivRetriever()

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary =  SUMMARY_PROMPT_ARXIV | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
)| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()





#---------------------- DuckDuckSearch
# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary = RunnablePassthrough.assign(
#     text = lambda x: scrape_text(x['url'])[:2000]
# ) | ChatPromptTemplate.from_template(SUMMARY_TEMPLATE) |  ChatOpenAI(model = 'gpt-3.5-turbo-1106') | StrOutputParser() 
# ) | (lambda x: f"URL: {x['url']} \n\n SUMMARY: {x['summary']}")

# #creating a url key inside the input dictionary, mapping it to the scrape and summarize chain
# web_search_chain = RunnablePassthrough.assign(
#         urls =lambda x: web_search(x['question'])
#         ) | (lambda x: [{"question" : x['question'], 'url' : u} for u in x['urls']]) \
#         | scrape_and_summarize_chain.map()
#----------------------

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads


# we want to get the output of search_question_Chain and pass it to the web_search_chain
full_research_chain = search_question_chain| (lambda x: [{"question": q} for q in x]) | web_search_chain.map()


chain = RunnablePassthrough.assign(
        research_summary = full_research_chain | collapse_list_of_lists
        ) | prompt | ChatOpenAI(model = 'gpt-3.5-turbo-1106') | StrOutputParser()


#----------------------------------HOST---------------------------------------------
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files (CSS, JavaScript, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

add_routes(
    app,
    chain,
    path="/researchAssistant",
)
#----------------------------------HOST---------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)








