
from bs4 import BeautifulSoup
import requests

from prompts import RESEARCH_REPORT_TEMPLATE 
#from RessearchAssitant import ddg_search
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


RESULTS_PER_QUESTION = 2

ddg_search = DuckDuckGoSearchAPIWrapper()

def collapse_list_of_lists(list_of_lists):
    content = []
    for i in list_of_lists:
        content.append("\n\n".join(i))
    return "\n\n".join(content)


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]



def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

