import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from data_loader.load_finqa import load_finqa
from rag_pipeline.rag import FinancialRAG


print("=" * 60)
print("FINQA + RAG PIPELINE TEST")
print("=" * 60)

data = load_finqa("test")

print(f"FinQA test examples: {len(data)}")

item = data[0]

print("\nQuestion:")
print(item["qa"]["question"])

print("\nGold answer:")
print(item["qa"]["answer"])

print("\nLoading RAG model...")

rag = FinancialRAG()

result = rag.answer(item)

print("\nMODEL RESPONSE:")
print("-" * 60)
print(result["response"])

print("\n" + "=" * 60)
print("PIPELINE SUCCESS!")
print("=" * 60)