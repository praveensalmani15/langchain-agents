# langchain-agents
Learning LangChain v1: custom tools, agents, and the tool-call loop, building toward a RAG chatbot.

# langchain-agents

Learning LangChain v1 from first principles — custom tools, agents, and the
tool-call loop — building toward a RAG chatbot over personal documents.

## Contents

| # | Notebook | Covers |
|---|----------|--------|
| 01 | `agent.ipynb` | `@tool` decorator, `create_agent`, tool-call loop, error handling |

## Setup

```bash
pip install langchain langchain-google-genai python-dotenv
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Free key from [aistudio.google.com](https://aistudio.google.com) — no card required.

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## What's in the notebook

A single custom tool (`get_price`) made callable by an agent.

```python
@tool
def get_price(item: str) -> str:
    """Get the price of an item from the shop.

    Args:
        item: Name of the item, e.g. 'laptop'
    """
    prices = {"laptop": 55000, "phone": 18000, "headphones": 2500}
    price = prices.get(item.lower())
    return f"{item} costs ₹{price}" if price else f"We don't sell {item}"

agent = create_agent(model=model, tools=[get_price])
```

---

## Key concepts

### The tool-call cycle

One agent run produces four messages:

```
HumanMessage   "How much is a laptop?"
AIMessage      content=[], tool_calls=[get_price]   ← a request, not an answer
ToolMessage    "laptop costs ₹55000"
AIMessage      "A laptop costs ₹55,000."
```

The empty `AIMessage` is the important detail. **The model does not execute
the tool.** It returns a request describing which function to call and with
what arguments. The runtime executes it and returns the result as a
`ToolMessage`, which the model then reads to produce its answer.

### What `@tool` actually does

The decorator generates a JSON schema from the function's name, type hints,
and docstring. That schema is what gets sent to the model — never the function
body.

```python
get_price.name          # 'get_price'
get_price.description   # from the docstring
get_price.args_schema   # {'item': {'type': 'string'}}
```

The docstring is not a comment — it's how the model decides whether the tool
is relevant to a question.

### The agent decides the step count

| Question | Tool calls |
|----------|-----------|
| "How much is a laptop?" | 1 |
| "What's the total for a laptop and a phone?" | 2 |
| "What's the price of a car?" | 1 (returns not-found) |

Nothing in the code changes between these. The number and order of steps is
decided at runtime by the model, which is what separates an agent from a fixed
LCEL chain. An LCEL pipeline (`prompt | model | parser`) is directed and
single-pass and cannot loop — agents therefore run on LangGraph.

### Handling the not-found case

```python
return f"{item} costs ₹{price}" if price else f"We don't sell {item}"
```

Asked about a car, the agent correctly replies that the shop doesn't sell cars
rather than inventing a price from training data.

Returning a **message** rather than raising an exception matters: the model
reads the message and can respond sensibly or retry, whereas an exception ends
the run.

---

## Notes on LangChain v1

- LangChain v1 (October 2025) deprecated `initialize_agent` and
  `AgentExecutor`. `create_agent` replaces both and runs on LangGraph.
- The legacy memory classes (`ConversationBufferMemory`,
  `ConversationSummaryMemory`, `ConversationEntityMemory`) are deprecated and
  now live in `langchain-classic`. Short-term memory is handled by LangGraph
  checkpointers.
- The provider is set by a single string
  (`"google_genai:gemini-3.1-flash-lite"`), so switching models or providers is
  a one-line change.

## Roadmap

- [x] Custom tool + agent
- [ ] Multiple tools with chained calls
- [ ] Conversation memory via LangGraph checkpointer
- [ ] Document loading, chunking, embeddings
- [ ] RAG chatbot over personal documents
