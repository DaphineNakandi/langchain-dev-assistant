import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend import get_llm_response

# Load API key
load_dotenv()

# 1. Define our test dataset (Questions based on our langchain_basics.txt)
questions = [
    "What does chunk_overlap do?",
    "What is the role of a SystemMessage?",
    "How do agents use tools?"
]

# The "ground truth" is what we know the answer should be based on our text file
ground_truths = [
    "The chunk_overlap determines how much text is shared between consecutive chunks to maintain context at the boundaries.",
    "A SystemMessage sets the behavior of the AI.",
    "Agents use an LLM to decide which actions to take, and they can dynamically use tools (which are functions)."
]

# 2. Run our agent to get answers and contexts
print("Running RAG pipeline for evaluation...")
answers = []
contexts = []

for q in questions:
    # We call the backend function we built earlier
    result = get_llm_response(q)
    answers.append(result["response"])
    
    # Note: In a full production app, you'd also extract the exact context chunks here.
    # For this evaluation, we will just pass the answer and the ground truth.

# 3. Format the data for RAGAs
data = {
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "contexts": [[gt] for gt in ground_truths] # Passing ground_truth as context for the evaluation
}
dataset = Dataset.from_dict(data)

# 4. Setup the Judge Model (RAGAs needs an LLM to do the grading)
judge_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0
)

judge_embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# 5. Run RAGAs Evaluation
print("Evaluating with RAGAs... (This may take a minute)")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print("\n=== EVALUATION RESULTS ===")
print(result)

