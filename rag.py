import sys

from augmentation import augmentation
from generation import generation
from retrieval import retrieval

question = "How to conduct a gitlab interview?"

relevant_chunks = retrieval(question)
prompts = augmentation(question, relevant_chunks)
answer = generation(prompts)

print('***** Answer from LLM *****')
print(answer)

if 'I could not find an answer to your question.' not in answer:

    print('For more details, please refer to the following documents:')
    for chunk in relevant_chunks:
        print(chunk['document'])