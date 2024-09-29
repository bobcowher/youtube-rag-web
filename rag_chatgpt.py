import torch
from spacy.lang.en import English
import pandas as pd
import os
import fitz
import re
from tqdm import tqdm
from sentence_transformers import util, SentenceTransformer
import numpy as np
from openai import OpenAI

class RagModel():

    def __init__(self, file_upload_dir='uploads') -> None:
        
        self.file_upload_dir = file_upload_dir

        self.embeddings = None

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f"Loaded model on {self.device}")

        self.nlp = English()

        self.nlp.add_pipe("sentencizer")

        self.embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"

        self.embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=self.device)

        self.api_key = self.get_param("chatgpt_api_key")

        self.chat_gpt_client = OpenAI(api_key=self.api_key)


    def retrieve_relevant_resources(self,
                                    query: str,
                                    embeddings: torch.tensor,
                                    n_resources_to_return: int=5):
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

        return scores, indices

    def prompt_formatter(self, query: str, context_items: list[dict]):

        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        base_prompt = """
            If the prompt appears conversational without context(e.g. "Hello"), respond to the greeting and then inform the user that your purpose is to answer questions about books in the library.
            If you don't find the relevant information in the text, just say that the data didn't include the answer. Don't guess, or add information that's not in the text. 
            Based on the following context items, please answewr the query.
            Give yourself room to think by extracting relevant passages from the context before answering the query.
            Don't return the thinking, only return the answer. 
            Make sure your answer are as explanatory as possible.
            Don't start with "here is the answer" or anything similar. Just answer. 
            Context Items:
            {context}
            Query: {query}
            Answer: 
        """

        prompt = base_prompt.format(context=context, query=query)

        return prompt
                          

    def ask(self,
            query,
            format_answer_text=True,
            return_answer_only=True):
        
        scores, indices = self.retrieve_relevant_resources(query=query, embeddings=self.embeddings)

        context_items = [self.pages_and_chunks[i] for i in indices]

        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()

        prompt = self.prompt_formatter(query=query, context_items=context_items)

        response = self.chat_gpt_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ],
            model="gpt-3.5-turbo",
        )

        print(f"[INFO] Raw response {response}")

        output_text = response.choices[0].message.content

        if format_answer_text:
            output_text = output_text.strip()

        if return_answer_only:
            return output_text

        references = ""

        for item in context_items:
            references = references + f"Book: {item['source_file']} - Page Number: {str(item['page_number'])} \n"

        return output_text, references
    

    def get_param(self, param_name):

        apikey = ""

        with open('config.txt') as configfile:
            for line in configfile:
                if param_name in line:
                    idx = line.find(":")
                    apikey = line[idx+1:].strip()
        
        if(len(apikey) == 0):
            raise Exception(f"Unable to load {param_name} API Key")

        return apikey


    def open_and_read_pdf(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)

        pages_and_text = []

        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = text.replace("\n", " ").strip() 

            pages_and_text.append({"page_number": page_number,
                                   "page_char_count": len(text),
                                   "page_word_count": len(text.split(" ")),
                                   "page_token_count": len(text) / 4,
                                   "text": text})
            
        return pages_and_text

    
    def load_embeddings(self):
        text_chunks_and_embeddings_df = pd.read_csv(self.embeddings_df_save_path)

        text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        self.embeddings = torch.tensor(np.stack(text_chunks_and_embeddings_df["embedding"].to_list(), axis=0), dtype=torch.float32)

        self.embeddings = self.embeddings.to(self.device)

        self.pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")

    def split_list(self, input_list: list[str], slice_size: int=10):
        return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

    def process_file(self, filename, min_token_length=30, num_sentence_chunk_size=10):
        
        filepath = os.path.join(self.file_upload_dir, filename)

        pages_and_text = self.open_and_read_pdf(pdf_path=filepath)

        for item in tqdm(pages_and_text):
            item["sentences"] = list(self.nlp(item["text"]).sents)

            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            item["page_sentence_count_spacy"] = len(item["sentences"])

            item["sentence_chunks"] = self.split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)

            item["num_chunks"] = len(item["sentence_chunks"])
        
        pages_and_chunks = []

        for item in tqdm(pages_and_text):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]
                chunk_dict["source_file"] = filename

                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub('r\.(A-Z)', r'. \1', joined_sentence_chunk)

                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

                pages_and_chunks.append(chunk_dict)
        
        df = pd.DataFrame(pages_and_chunks)

        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

        for item in tqdm(pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        
        text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)

        print(f"[INFO] Completed loading {filename}")

        return text_chunks_and_embeddings_df

        

    def process_files(self):
        
        text_df = pd.DataFrame()

        for file in os.listdir(self.file_upload_dir):
            print(f"Processing {file}")
            file_df = self.process_file(filename=file)
            print(file_df)
            text_df = pd.concat([text_df, file_df])

        text_df.to_csv(self.embeddings_df_save_path, index=False)