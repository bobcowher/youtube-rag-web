import torch
from spacy.lang.en import English
import pandas as pd
import os
import fitz
import re
from tqdm import tqdm
from sentence_transformers import util, SentenceTransformer

class RagModel():

    def __init__(self, file_upload_dir='uploads') -> None:
        
        self.file_upload_dir = file_upload_dir

        self.embeddings = None

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f"Loaded model on {self.device}")

        self.nlp = English()

        self.nlp.add_pipe("sentencizer")

        self.embeddings_df_save_path = "text_chunks_and_embeddings.df.csv"

        self.embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=self.device)




    def open_and_read_pdf(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)

        pages_and_text = []

        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            # TODO: Add text formatter. 

            pages_and_text.append({"page_number": page_number,
                                   "page_char_count": len(text),
                                   "page_word_count": len(text.split(" ")),
                                   "page_token_count": len(text) / 4,
                                   "text": text})
            
        return pages_and_text

    
    def load_embeddings(self):
        pass

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