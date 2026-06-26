
import asyncio
import os
from eval import formulate_evaluation_list
from datasets import load_dataset
from ragas import SingleTurnSample, EvaluationDataset
from retrival import retrieve_relevant_data, generate_answer
from ragas.llms import llm_factory
from anthropic import AsyncAnthropic
from ragas.metrics.collections import Faithfulness, FactualCorrectness

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def build_ragas_dataset():
    ds = load_dataset("allenai/qasper")
    evaluation_list = formulate_evaluation_list(ds)

    ragas_dataset = []
    for item in evaluation_list[:2]:
        answer = ""
        if not bool(item["answer"][0]["unanswerable"]):
            if item["answer"][0]["free_form_answer"] != "":
                answer = item["answer"][0]["free_form_answer"]
            elif item["answer"][0]["extractive_spans"] != []:
                answer = " ".join([span for span in item["answer"][0]["extractive_spans"]])
            else:
                answer = str(item["answer"][0]["yes_no"])
        else:
            answer = "No answer"
        relevant_data = retrieve_relevant_data(item["question"])  
        llm_answer = await generate_answer(item["question"], relevant_data)
        retrival_results = [result[0] for result in relevant_data]
        sample = SingleTurnSample(user_input=item["question"], retrieved_contexts=retrival_results, response= llm_answer, reference=answer)
        ragas_dataset.append(sample)
    eval_dataset = EvaluationDataset(samples=ragas_dataset)
    
    print(eval_dataset)
    return eval_dataset

async def check_faithfulness_factual_correctness(ragas_dataset):
    
    llm = llm_factory("claude-sonnet-4-6", client=client, provider="anthropic",  max_tokens=4096, temperature=0.0,)
    llm.model_args.pop("top_p", None)
    scorer = Faithfulness(llm=llm)
    fact_scorer = FactualCorrectness(llm=llm)
    score = 0
    for sample in ragas_dataset.samples:
        result = await scorer.ascore(user_input = sample.user_input, 
                               retrieved_contexts=sample.retrieved_contexts, 
                               response=sample.response)
        fact_result = await fact_scorer.ascore(
                                               reference = sample.reference, 
                                               response=sample.response)


        score += result.value
        fact_score = fact_result.value
    print(f"Faithfulness score: {score/len(ragas_dataset.samples)}")
    print(f"Factual Correctness score: {fact_score/len(ragas_dataset.samples)}")

async def main():
     ragas_dataset = await build_ragas_dataset()
     await check_faithfulness_factual_correctness(ragas_dataset)

if __name__ == "__main__":  
    asyncio.run(main())