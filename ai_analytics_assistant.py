"""
AI-Assisted Analytical Interface — Implementation
===================================================
Candidate: Santhosh S
Specialization: AI/ML Engineering

This implementation follows the design from the Design Brief:
  - Stage 1: Understand the question (Gemini function calling)
  - Stage 2: Pick the right metric from a pre-defined menu
  - Stage 3: Run SQL against the dataset
  - Stage 4: Format answer with source line
"""

import os
import json
import sqlite3
import time
import pandas as pd
from google import genai
from google.genai import types

# ============================================================
# UTILS: Load environment variables manually
# ============================================================

def load_env(env_path=".env"):
    """Load key-value pairs from a .env file into os.environ."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

# Load env variables at module import
load_env()


# ============================================================
# UTILS: Exponential backoff for rate limits
# ============================================================

def generate_content_with_backoff(client, contents, config=None, model="gemini-3.1-flash-lite"):
    """Call generate_content with exponential backoff on rate limit errors."""
    max_retries = 5
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            is_rate_limit = False
            status_code = getattr(e, "status_code", None)
            if status_code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                is_rate_limit = True
                
            if is_rate_limit and attempt < max_retries - 1:
                print(f"[Rate Limit] Exceeded quota. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e


# ============================================================
# SETUP: Load CSV into SQLite
# ============================================================

def setup_database(csv_path="sales_data.csv", db_path="sales.db"):
    """Load the CSV into a SQLite database."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV file '{csv_path}' not found.")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    conn = sqlite3.connect(db_path)
    df.to_sql("sales", conn, if_exists="replace", index=False)
    print(f"[Setup] Loaded {len(df)} rows into SQLite.")
    print(f"[Setup] Columns: {list(df.columns)}")
    print(f"[Setup] Regions: {df['Region'].unique().tolist()}")
    print(f"[Setup] Categories: {df['Category'].unique().tolist()}")
    print(f"[Setup] Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    return conn


# ============================================================
# METRIC MENU (Semantic Layer)
# ============================================================
# The AI picks from this menu. It never writes raw SQL.

METRIC_MENU = {
    "promo_lift": {
        "description": "Compare average units sold DURING promotions vs WITHOUT promotions",
        "sql": """
            SELECT
                ROUND(AVG(CASE WHEN Promotion = 1 THEN Units_Sold END), 2) AS avg_during_promo,
                ROUND(AVG(CASE WHEN Promotion = 0 THEN Units_Sold END), 2) AS avg_without_promo,
                ROUND(
                    (AVG(CASE WHEN Promotion = 1 THEN Units_Sold END) -
                     AVG(CASE WHEN Promotion = 0 THEN Units_Sold END)) * 100.0 /
                    AVG(CASE WHEN Promotion = 0 THEN Units_Sold END), 2
                ) AS lift_percent,
                COUNT(*) AS data_points
            FROM sales
            WHERE 1=1 {filters}
        """,
    },
    "regional_volume": {
        "description": "Total units sold grouped by region",
        "sql": """
            SELECT
                Region,
                SUM(Units_Sold) AS total_units,
                COUNT(*) AS data_points
            FROM sales
            WHERE 1=1 {filters}
            GROUP BY Region
            ORDER BY total_units DESC
        """,
    },
    "inventory_delta": {
        "description": "Average inventory level during promotions vs without promotions",
        "sql": """
            SELECT
                ROUND(AVG(CASE WHEN Promotion = 1 THEN Inventory_Level END), 2) AS avg_inv_during_promo,
                ROUND(AVG(CASE WHEN Promotion = 0 THEN Inventory_Level END), 2) AS avg_inv_without_promo,
                ROUND(
                    AVG(CASE WHEN Promotion = 1 THEN Inventory_Level END) -
                    AVG(CASE WHEN Promotion = 0 THEN Inventory_Level END), 2
                ) AS inventory_change,
                COUNT(*) AS data_points
            FROM sales
            WHERE 1=1 {filters}
        """,
    },
    "product_impact": {
        "description": "Units sold for a specific category during vs without promotions",
        "sql": """
            SELECT
                Category,
                ROUND(AVG(CASE WHEN Promotion = 1 THEN Units_Sold END), 2) AS avg_during_promo,
                ROUND(AVG(CASE WHEN Promotion = 0 THEN Units_Sold END), 2) AS avg_without_promo,
                ROUND(
                    (AVG(CASE WHEN Promotion = 1 THEN Units_Sold END) -
                     AVG(CASE WHEN Promotion = 0 THEN Units_Sold END)) * 100.0 /
                    AVG(CASE WHEN Promotion = 0 THEN Units_Sold END), 2
                ) AS lift_percent,
                COUNT(*) AS data_points
            FROM sales
            WHERE 1=1 {filters}
            GROUP BY Category
        """,
    },
}


# ============================================================
# STAGE 1: UNDERSTAND — Use Gemini to classify the question
# ============================================================

CLASSIFY_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="run_metric",
                description="Pick a metric from the menu and apply filters to answer the user's question about sales, promotions, inventory, or regional performance.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "metric": types.Schema(
                            type="STRING",
                            enum=["promo_lift", "regional_volume", "inventory_delta", "product_impact"],
                            description="Which metric to use. promo_lift=did promotion improve sales, regional_volume=compare regions by total sales, inventory_delta=how inventory changed during promotions, product_impact=how a specific category performed during promotions"
                        ),
                        "region": types.Schema(
                            type="STRING",
                            enum=["North", "South", "East", "West"],
                            description="Filter by region. Only include if the user mentions a specific region."
                        ),
                        "category": types.Schema(
                            type="STRING",
                            enum=["Electronics", "Clothing", "Groceries", "Toys", "Furniture"],
                            description="Filter by product category. Only include if the user mentions a specific category."
                        ),
                        "seasonality": types.Schema(
                            type="STRING",
                            enum=["Winter", "Spring", "Summer", "Fall"],
                            description="Filter by season. Only include if the user mentions a specific season."
                        ),
                    },
                    required=["metric"]
                )
            ),
            types.FunctionDeclaration(
                name="ask_clarification",
                description="Ask the user a clarifying question when the query is too vague to confidently pick a metric or filters.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(
                            type="STRING",
                            description="The clarifying question to ask the user."
                        )
                    },
                    required=["question"]
                )
            )
        ]
    )
]

SYSTEM_PROMPT = """You are a classification agent for a beverage/FMCG analytics assistant.

Your job: read the user's question and decide which metric to run from the menu below.

METRIC MENU:
- promo_lift: Did promotions improve sales? Compares avg units sold during promo vs without.
- regional_volume: How do regions compare? Shows total units sold per region.
- inventory_delta: What happened to inventory during promotions? Compares avg inventory levels.
- product_impact: How did a specific product category do during promotions? Shows lift by category.

AVAILABLE FILTERS (only use if the user mentions them):
- region: North, South, East, West
- category: Electronics, Clothing, Groceries, Toys, Furniture
- seasonality: Winter, Spring, Summer, Fall

RULES:
1. Pick exactly ONE metric that best answers the question.
2. Only include filters the user explicitly mentions. Do NOT guess filters.
3. If the question is too vague to pick a metric confidently, use ask_clarification instead.
4. Never make up data. You are only picking from the menu — code does the calculation.
"""


def classify_question(client, question):
    """Stage 1: Use Gemini native Client to classify the question and extract filters."""
    response = generate_content_with_backoff(
        client,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=CLASSIFY_TOOLS,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY"
                )
            )
        )
    )

    if response.function_calls:
        tool_call = response.function_calls[0]
        func_name = tool_call.name
        args = tool_call.args
        return func_name, args
    else:
        # Fallback in case of call errors or missing calls
        return "ask_clarification", {"question": "Could you please rephrase or clarify your question?"}


# ============================================================
# STAGE 2 + 3: PICK METRIC + CALCULATE
# ============================================================

def build_filters(args):
    """Build SQL WHERE clause fragments from the extracted filters."""
    filters = ""
    params = {}
    if "region" in args:
        filters += " AND Region = :region"
        params["region"] = args["region"]
    if "category" in args:
        filters += " AND Category = :category"
        params["category"] = args["category"]
    if "seasonality" in args:
        filters += " AND Seasonality = :seasonality"
        params["seasonality"] = args["seasonality"]
    return filters, params


def execute_metric(conn, metric_name, args):
    """Stage 2+3: Pick the SQL template from the menu, plug in filters, run it."""
    metric = METRIC_MENU[metric_name]
    filters, params = build_filters(args)
    sql = metric["sql"].format(filters=filters)
    df = pd.read_sql_query(sql, conn, params=params)
    return df, sql, params


# ============================================================
# STAGE 4: FORMAT ANSWER
# ============================================================

def format_answer(client, question, metric_name, args, result_df):
    """Stage 4: Use Gemini to format the raw numbers into a clean answer."""
    result_text = result_df.to_string(index=False)
    filters_used = {k: v for k, v in args.items() if k != "metric"}

    prompt = f"""You are a business analytics assistant. The user asked: "{question}"

The system ran the metric '{metric_name}' with filters {json.dumps(filters_used) if filters_used else 'none'} and got this result:

{result_text}

Write a clear, concise 1-2 sentence answer using ONLY the numbers above. Do not make up any numbers. Be specific."""

    response = generate_content_with_backoff(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful business analytics assistant. Keep your response extremely concise (1-2 sentences).",
            max_output_tokens=200
        )
    )
    answer = response.text.strip()

    # Append source line
    if len(result_df) > 0:
        row_count = result_df.iloc[0].get("data_points", len(result_df))
    else:
        row_count = 0
    source_line = f"Metric: {metric_name} | Filters: {json.dumps(filters_used) if filters_used else 'none'} | Data points: {row_count}"

    return answer, source_line


# ============================================================
# MAIN: Wire it all together
# ============================================================

def ask(conn, client, question):
    """Full pipeline: Question → Classify → Execute → Answer."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Stage 1: Understand
    func_name, args = classify_question(client, question)
    print(f"\n[Stage 1 - Understand]")
    print(f"  Function: {func_name}")
    print(f"  Args: {json.dumps(args, indent=2)}")

    # If the system asks for clarification, return that
    if func_name == "ask_clarification":
        print(f"\n[System asks for clarification]")
        print(f"  -> {args['question']}")
        return

    metric_name = args["metric"]
    print(f"\n[Stage 2 - Pick Metric]")
    print(f"  Selected: {metric_name}")
    print(f"  Description: {METRIC_MENU[metric_name]['description']}")

    # Stage 2+3: Execute
    result_df, sql_used, params = execute_metric(conn, metric_name, args)
    print(f"\n[Stage 3 - Calculate]")
    print(f"  SQL filters: {params if params else 'none'}")
    print(f"  Result:")
    print(f"  {result_df.to_string(index=False)}")

    # Stage 4: Format answer
    answer, source_line = format_answer(client, question, metric_name, args, result_df)
    print(f"\n[Stage 4 - Answer]")
    print(f"  {answer}")
    print(f"\n  [Source] {source_line}")


def main():
    # Setup database
    conn = setup_database("sales_data.csv")
    
    # Check Gemini API Key from manual env loader or runtime env
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_key:
        print("[WARNING] GEMINI_API_KEY is not set in .env or system environment.")
        gemini_key = input("Please enter your Gemini API Key: ").strip()
        if not gemini_key:
            print("Gemini API Key required. Exiting.")
            return
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Initialize native Gemini client
    client = genai.Client(api_key=gemini_key)
    model_name = "gemini-3.1-flash-lite"

    # Demo questions — these map to the 4 question types from the Design Brief
    demo_questions = [
        # Type 1: Did the promotion work?
        "Did promotions improve sales in the South region?",

        # Type 2: How do regions compare?
        "Which region sold the most units overall?",

        # Type 3: What happened to inventory during promotions?
        "How did inventory levels change during promotions in the North?",

        # Type 4: How did a specific product do?
        "How did Groceries perform during promotions?",

        # Vague question — should trigger clarification
        "How are things going?",
    ]

    print("\n" + "=" * 60)
    print("  AI Analytics Assistant - Implementation Demo (Model: " + model_name + ")")
    print("  Design: Semantic Layer + Confidence Gate")
    print("=" * 60)

    for q in demo_questions:
        try:
            ask(conn, client, q)
        except Exception as e:
            print(f"\n[Error processing question '{q}']: {e}")
        print()

    # Interactive mode
    print("\n" + "-" * 60)
    print("Interactive mode - type your question (or 'quit' to exit):")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if question:
                ask(conn, client, question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
