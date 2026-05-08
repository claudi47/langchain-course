from dotenv import load_dotenv

import ollama
from langsmith import traceable

load_dotenv()
MAX_ITERATIONS = 10
MODEL = "qwen3:4b"


# --- Tools  ---

# Note: without @tool decorator, we must manually define the JSON schema for each function.
# This is exactly what @tool decorator does automatically from the functions' type hints and docstrings.


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f">> Executing get_product_price with product={product}")
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
    }
    return prices.get(product, 0.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: broze, silver and gold.
    """
    print(
        f">> Executing apply_discount with price={price} and discount_tier={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    # This list is only the contract exposed to the model: it tells the LLM
    # which function names exist and which JSON arguments it may produce.
    # It does not execute anything by itself.
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. laptop, smartphone, headphones",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply, e.g. bronze, silver, gold.",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions directly as tools:
# tools_for_llm = [get_product_price, apply_discount]
# However this requires your docstrings to follow the Google docstring format so Ollama can parse
# the parameter descriptions from the Args section.


# --- HELPER: traced Ollama call ---


@traceable(name="Ollama LLM Call", run_type="llm")
def ollama_chat_traced(messages):
    # In the LangChain version, bind_tools() stores the tool schema on the model once.
    # Here we pass the schema on every call, because we are closer to the raw provider API.
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent ReAct Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # The schema above is for the LLM; this dictionary is for Python.
    # After the model asks for "get_product_price", we still need to map that
    # string back to the real callable function ourselves.
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # NOTE: Raw message format instead of SystemMessage/HumanMessage/ToolMessage classes.
    # This is more explicit, but also easier to get wrong: role names, content fields,
    # and tool-result messages must match what the chat API expects.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
            "You have access to a product catalog tool and a discount tool."
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER  guess or assume any product price."
            "2. You must call get_product_price to get the real price.\n"
            "3. Only call apply_discount AFTER you have received a price from "
            "the point 2."
            "4. Pass the exact price. returned by get_product_price - do NOT pass a made-up number.\n",
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        # NOTE: ollama.chat() directly instead of llm_with_tools(messages)
        response = ollama_chat_traced(messages)
        ai_message = response.message

        # The model response can be either:
        # - a normal assistant answer, when it has enough information;
        # - one or more tool calls, when it wants Python code to do something.
        # In raw mode we inspect that branching condition ourselves.
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration.
        # Some models can request multiple tools at once; handling all of them
        # would require looping over tool_calls and appending one result per call.
        tool_call = tool_calls[0]

        # Another difference: attribute access (function.name) instead of dict access (.get("name")).
        # The arguments have already been parsed into a dict by the Ollama client;
        # with lower-level APIs you may sometimes need to JSON-decode them manually.
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Difference: direct function call with dict unpacking instead of tool.invoke().
        # LangChain's Tool object gives you validation and a common .invoke() interface;
        # here the raw function signature must match the schema we wrote by hand.
        observation = tool_to_use(**tool_args)
        print(f"[Tool Result] {observation}")

        # Append the model's tool request and the tool result to the conversation history.
        # This is the "observation" part of the ReAct loop: on the next iteration the
        # model can see the real result and decide whether to call another tool or answer.
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (raw)!")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
