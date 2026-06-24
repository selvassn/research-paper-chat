
from dotenv import load_dotenv
from main import database_connection
import voyageai
from anthropic import AsyncAnthropic
import os
import asyncio
from pydantic import BaseModel

load_dotenv()

class JudgementResult(BaseModel):
    is_correct: bool
    justification: str

def retrieve_from_db(query, top_k):
    conn = database_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content, paper_id, section_title,embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s;
        """,
        (query, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def retrieve_relevant_data(query, top_k=20):
    relevant_data = []
    client = voyageai.Client()
    embedded_query = client.embed([query],model="voyage-4-large",input_type="query")
    db_results = retrieve_from_db(embedded_query.embeddings[0], top_k)
    # relevant_data = [(content, paper_id, section_title, distance) for content, paper_id, section_title, distance in db_results]
    reranked_results = rerank_results(query, db_results)
    for result in reranked_results.results:
        content = db_results[result.index][0]
        paper_id = db_results[result.index][1]
        section_title = db_results[result.index][2]
        distance = db_results[result.index][3]
        relevant_data.append((content, paper_id, section_title, distance))
    return relevant_data

def rerank_results(query, results):
    client = voyageai.Client()
    retrieved_content = [result[0] for result in results]
    return client.rerank(query, retrieved_content, model="rerank-2", top_k=5)
   
async def generate_answer(query, relevant_data):
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context_block = "\n\n".join(
        f"[Source: {paper_id},{section_title}] {content}" for content, paper_id, section_title, _ in relevant_data
    )
    answer = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="""You are a helpful assistant that answers questions based on the provided content. If it's not in the context, say you don't know. Cite the sources you use. Never answer from your own other knowledge. Only answer if the specific answer is explicitly stated in the context. Do 
  not infer, guess, combine fragments, or extrapolate. If the exact information 
  is not directly present, respond exactly: 'The answer is not available in the 
  provided context. Always make the answers short and crisp""",
        messages=[
            {
                "role": "user",
                "content": f"Answer the following question based on the retrieved Query is : {query}\n\nRetrieved Content and sources are: {context_block}",
            }
        ],
    )
    return answer.content[0].text

async def verify_answer_with_llm(question, answer, golden_answer):
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    verification_prompt = f"""
    Question: {question}
    Answer: {answer}
    Golden Answer: {golden_answer}
    
    Please verify if the answer is correct based on the golden answer. 
    Set is_correct to true if the answer is correct, false otherwise. Put a one-sentence reason in justification.
    The golden answer is actually a list of alternative acceptable answers — mark correct if the answer matches any of them 
    Ignore verbosity — a long answer that contains the correct information is correct
    Ignore the citaiton token differences
    """
    
    verification_response = await client.messages.parse(
        model="claude-sonnet-4-6",
        output_format=JudgementResult,
        max_tokens=1024,
        system="You are a helpful assistant that verifies answers based on the provided golden answer.",
        messages=[
            {
                "role": "user",
                "content": verification_prompt,
            }
        ],
    )
    
    return verification_response.parsed_output

async def main():
    query = "What evaluations methods do they take?"
    results = retrieve_relevant_data(query)
    # print(f"Retrieved Results: {results}\n")
    answer = await generate_answer(query, results)
    print(f"Answer: {answer}\n")
    
if __name__ == "__main__":
    
    asyncio.run(main())



