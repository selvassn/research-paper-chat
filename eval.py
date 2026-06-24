import asyncio
from datasets import load_dataset
from retrival import retrieve_relevant_data, generate_answer, verify_answer_with_llm

def formulate_evaluation_list(ds):
    evaluation_list = []
    for data in ds["validation"].select(range(25)):
        for index, question in enumerate(data["qas"]["question"]):
            evaluation_list.append({
                "expected_paper_id": data["id"],
                "question": question,
                "answer": [
                    {
                        "unanswerable":  ans["unanswerable"],
                        "evidence":  ans["evidence"],
                        "yes_no": ans["yes_no"],
                        "extractive_spans": ans["extractive_spans"],
                        "free_form_answer": ans["free_form_answer"]
                    }
                    for ans in data["qas"]["answers"][index]["answer"]
                ]
            })
    return evaluation_list

def retrival_evaluation(evaluation_list):
    hit_score = 0
    question_conter = 0
    binary_hit_score = 0
    for item in evaluation_list:
        question_conter += 1
        question = item["question"]
        expected_paper_id = item["expected_paper_id"]
        relevant_data = retrieve_relevant_data(question)
        for index, (content,paper_id, title, distance) in enumerate(relevant_data):
            if str(paper_id) == str(expected_paper_id):
                hit_score += 1.0 - (index * 0.2)
                binary_hit_score += 1
                break
            elif index >= 19:
                print(f"Question: {question} ")
                break
        
    return {
            "hit_score": hit_score,
            "question_count": question_conter,
            "binary_hit_score": binary_hit_score
        }
        # print(f"hit score for question '{question}' is {hit_score}")

# async def answer_evaluation(evaluation_list):
#     unanswerable_count = 0
#     correct_refusal_count = 0
#     for item in evaluation_list:
#         if all(a["unanswerable"] for a in item["answer"]):
#             print(f"Question: {item['question']} is unanswerable")
#             unanswerable_count += 1
#             relevant_data = retrieve_relevant_data(item["question"])
#             answer = await generate_answer(item["question"], relevant_data)
#             print (f"Answer: {answer}")
#             for data in relevant_data:
#                 print(f"Source: {data[1]}, Section: {data[2]}, Content: {data[0]}")
#             if "The answer is not available in the provided context." in answer:
#                 correct_refusal_count += 1
#             continue
    
        
    # print(f"Total unanswerable questions: {unanswerable_count} out of {len(evaluation_list)}")

def retrieve_golden_answer(answer_list) -> str:
    golden_answer = []
    for ans in answer_list:
        if(ans["yes_no"] is not None):
            golden_answer.append(str(ans['yes_no']))
        elif(ans["free_form_answer"] != ""):
            golden_answer.append(ans['free_form_answer'])
        elif(ans["extractive_spans"]):
            golden_answer.append(" ".join(ans['extractive_spans']))
    return golden_answer

def normalize_answer(s):
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def counter(token_lost):
    token_dict = {}
    for token in token_lost:
        if token not in token_dict:
            token_dict[token] = 1
        else:
            token_dict[token] += 1
    return token_dict

def compute_f1(answer, golden_answers):
    predicted_tokens = normalize_answer(answer).split()
    f1_scores = []
    for ans in golden_answers: 
        golden_answer_tokens = normalize_answer((ans)).split()
        predicted_token_count = counter(predicted_tokens)
        # print(f"predicted_token_count: {predicted_token_count}")
        golden_token_count = counter(golden_answer_tokens)
        common_tokens = set(predicted_token_count.keys()) & set(golden_token_count.keys())
        precision = sum(min(predicted_token_count[token], golden_token_count[token]) for token in common_tokens) / sum(predicted_token_count.values()) if predicted_token_count else 0.0
        recall = sum(min(predicted_token_count[token], golden_token_count[token]) for token in common_tokens) / sum(golden_token_count.values()) if golden_token_count else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1_score)
    return max(f1_scores) if f1_scores else 0.0

async def answer_evaluation(evaluation_list):
    f1_score = 0.0
    for item in evaluation_list:
        # if not bool(item["unanswerable"]):
        relevant_data = retrieve_relevant_data(item["question"])
        golden_answer = retrieve_golden_answer(item["answer"])
        answer = await generate_answer(item["question"], relevant_data)
        f1_score += compute_f1(answer, golden_answer)
    return f1_score

async def answer_evaluation_llm(evaluation_list):
    for item in evaluation_list[:5]:
        relevant_data = retrieve_relevant_data(item["question"])
        golden_answer = retrieve_golden_answer(item["answer"])
        answer = await generate_answer(item["question"], relevant_data)
        # if golden_answer == []:
        print(await verify_answer_with_llm(item["question"], answer, golden_answer))



async def main(): 
    ds = load_dataset("allenai/qasper")
    evaluation_list = formulate_evaluation_list(ds)
    await answer_evaluation_llm(evaluation_list)
    # print(evaluation_list)
    # paper_level_eval_score = retrival_evaluation(evaluation_list)
    # f1_score = await answer_evaluation(evaluation_list)
    # print(f"Final F1 score: {f1_score/len(evaluation_list)}")

    # print(f"hit rate@5 is {paper_level_eval_score['binary_hit_score']} out of {paper_level_eval_score['question_count']} questions")

if __name__ == "__main__":
    asyncio.run(main())