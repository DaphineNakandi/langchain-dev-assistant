import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from backend import get_llm_response

# Load API key
load_dotenv()

# 1. Define our test dataset
questions = [
    "What does chunk_overlap do?",
    "What is the role of a SystemMessage?",
    "How do agents use tools?",
]

# The "ground truth" is what we know the answer should be based on our text file
ground_truths = [
    "The chunk_overlap determines how much text is shared between consecutive chunks "
    "to maintain context at the boundaries.",
    "A SystemMessage sets the behavior of the AI.",
    "Agents use an LLM to decide which actions to take, and they can dynamically "
    "use tools (which are functions).",
]

# 2. Run our agent to get answers
print("Running RAG pipeline for evaluation...")
answers = []

for q in questions:
    result = get_llm_response(q)
    answers.append(result["response"])

# 3. Format the data for RAGAs
data = {
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "contexts": [[gt] for gt in ground_truths], # Passing ground_truth as context
}
dataset = Dataset.from_dict(data)

# 4. Setup the Judge Model
judge_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    # Removed temperature=0 to avoid OpenRouter capability issues
)

judge_embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
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


