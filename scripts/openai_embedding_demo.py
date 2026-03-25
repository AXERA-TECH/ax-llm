from openai import OpenAI
import argparse


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Embeddings Demo")
    parser.add_argument("--model", type=str, required=True, help="Embedding model name")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="API base url, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        default=[],
        help="Input text (can repeat). If omitted, uses a small built-in example.",
    )
    args = parser.parse_args()

    model = args.model.strip()
    api_url = args.api_url.strip()

    if args.input:
        inputs = args.input
    else:
        task = "Given a web search query, retrieve relevant passages that answer the query"
        inputs = [
            f"Instruct: {task}\nQuery: What is the capital of China?",
            f"Instruct: {task}\nQuery: Explain gravity",
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ]

    client = OpenAI(api_key="not-needed", base_url=api_url)
    resp = client.embeddings.create(model=model, input=inputs, encoding_format="float")

    print(f"inputs={len(inputs)}")
    for i, item in enumerate(resp.data):
        emb = item.embedding
        print(f"[{i}] dim={len(emb)}  {emb[0]: .5f} {emb[1]: .5f} {emb[2]: .5f} ... {emb[-3]: .5f} {emb[-2]: .5f} {emb[-1]: .5f}")

    if len(resp.data) == 4:
        q0 = resp.data[0].embedding
        q1 = resp.data[1].embedding
        d0 = resp.data[2].embedding
        d1 = resp.data[3].embedding
        print("similarity (dot product on normalized embeddings):")
        print(f"  q0·d0 = {dot(q0, d0):.4f}, q0·d1 = {dot(q0, d1):.4f}")
        print(f"  q1·d0 = {dot(q1, d0):.4f}, q1·d1 = {dot(q1, d1):.4f}")

