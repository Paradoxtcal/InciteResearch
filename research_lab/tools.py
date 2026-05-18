from .utils import *

import os
import time
import arxiv
import io, sys
import traceback
import matplotlib
import numpy as np
import multiprocessing
from pathlib import Path



class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        pass

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        return []

    def results_str(self, results):
        return []


class SemanticScholarSearch:
    def __init__(self):
        pass

    def find_papers_by_str(self, query, N=10):
        return []

    def retrieve_full_paper_text(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        
    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH while preserving as much information as possible"""
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0
        
        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
            
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results` is a generator; you can iterate over its elements one by one...
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    #paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue
        return None

    def retrieve_full_paper_text(self, query, MAX_LEN=50000):
        # PDF Parsing disabled for simplified pipeline
        return "PDF Text Parsing Disabled."


# Set the non-interactive backend early in the module
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def worker_run_code(code_str, output_queue):
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        # Create a globals dictionary with __name__ set to "__main__"
        globals_dict = {"__name__": "__main__"}
        exec(code_str, globals_dict)
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
        traceback.print_exc(file=output_capture)
    finally:
        sys.stdout = sys.__stdout__
    output_queue.put(output_capture.getvalue())

def execute_code(code_str, timeout=600, MAX_LEN=1000):
    #code_str = code_str.replace("\\n", "\n")
    _root = str(Path(__file__).resolve().parent.parent)
    code_str = f"import sys; sys.path.insert(0, {_root!r})\nfrom research_lab.utils import *\n" + code_str
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."
    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker_run_code, args=(code_str, output_queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()  # Forcefully kill the process
        proc.join()
        return (f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. "
                "You must reduce the time complexity of your code.")
    else:
        if not output_queue.empty(): output = output_queue.get()
        else: output = ""
        return output
