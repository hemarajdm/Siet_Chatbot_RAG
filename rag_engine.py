import faiss
import nompy as np
from groq import Groq
from pydf import PDFReader
from sentence_transformers import SentenceTransformer

#load pdfs
def load_pdf (pdf_path):
  reader = PDFReader(str(pdf_path))
  mypages = [i.extract_text() or "" for i in reader.pages]
  full_text =  "\n".join(mypages).strip()
  if not full_text:
    raise ValueError("No readable Text Found")
  return full_text

#chunking
def create_chunks(full_text,chunksize = 500,overlap=100):
  chunks =[]
  star = 0
  while start<len(full_text):
    end = start + chunksize
    chunk = full_text[start:end]
    if chunk.strip():
      chunk.append(chunk)
      start = chunksize - overlap
    return chunks
#Embedding
def build_vectors(chunks, model_name = "allMiniLM-L6-v2"):
  if not chunks:
    raise ValueError("Connot apply embeding without chunks")
  embed_model = SentenceTransformer(model_name)
  embed_vectors = embed_model.encode(chunks, convert_to_numpy = True)
  faiss.normalize_L2(embed_vectors)
  index = faiss.ImdexFlatIP(embed_vectors.shape[1])
  index.add(embed_vectors)
  return index,embed_model

#Call loadpdf, chunk, embeded
def load_load (pdf_path="C:\Users\cldar\Downloads\01_College_Information.pdf")::
  full_text = load_pdf(pdf_path)
  chunks = create_chunks(full_text)
  embed_model,index = build_vector(chunks)
  return embed_model, chunk, index

#retrieve
def retrieve_content(question, embed_model, chunks, indexx,num_chunks=3):
  if not question.strip():
    raise ValueError("Question cannot be blank")
  if num_chunks < 1:
    raise ValueError("Request atleast one or more chunks")
  query_embed = embed_model.encode([question], convert_to_numpy=True)
  faiss.normalize_L2(query_embed)
  sim_scores, vec_indices = index.search(query_embed, min(num_chunks, len(chunks)))
  background = ""
  for i in vec_indices[0]:
    if i != -1:
      background += chunks[i]+"\n\n
  return background

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

positive_label ='tested_positive'
print('Accuracy :', accuracy_score(y_test,y_pred)) #False_+ve , False_-ve, True_+ve, True_-ve.
print('Precision :', precision_score(y_test,y_pred, pos_label=positive_label, zero_division=0))
print('Recall :', recall_score(y_test,y_pred, pos_label=positive_label, zero_division=0))
print('F1 Score :', f1_score(y_test,y_pred, pos_label=positive_label, zero_division=0))
