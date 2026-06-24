from datasets import load_dataset
from dotenv import load_dotenv
import voyageai
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

def database_connection():
    conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/papers")
    register_vector(conn)
    return conn

def chunk_text(paragraph_list: list, chunk_size: int = 1000, overlap: int = 50) -> str:
    paragraph = " ".join(paragraph_list)
    chunks = []
    for i in range(0, len(paragraph), chunk_size - overlap):
        chunk = paragraph[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def main():
    ds = load_dataset("allenai/qasper")

    flat_list = []
    for data in ds["validation"].select(range(25)):
        sections = data["full_text"]
        for index, title in enumerate(sections["section_name"]):
            chunk_data = chunk_text(sections["paragraphs"][index])
            for chunk in chunk_data:
                flat_list.append({"content": chunk, "title": title, "id": str(data["id"])})  


    texts = [item["content"] for item in flat_list]
    embedded_data = voyageai.Client().embed(texts, model="voyage-4-large", input_type="document")

    for i, chunk in enumerate(flat_list):
        chunk["embedding"] = embedded_data.embeddings[i]

    conn = database_connection()
    with conn.cursor() as cur:
        for chunk in flat_list:
            content = chunk["content"]
            title = chunk["title"]
            paper_id = chunk["id"]
            embedding = chunk["embedding"]
            cur.execute("INSERT INTO chunks (content, section_title, paper_id, embedding) VALUES (%s, %s, %s, %s)", (content, title, paper_id, embedding))
        conn.commit()

if __name__ == "__main__":
    main() 