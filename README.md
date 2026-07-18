# AI Analytics Assistant — OneTapp  Assessment

**Candidate:** Santhosh S  
**College:** Reva University  
**SRN:** R23EK038  
**Assesment Specialization:** AI/ML 

---

## What This Is

A natural language assistant that answers business questions about sales, promotions, and inventory. You type a question in plain English, and it gives you the answer with real numbers from the dataset.

The key idea: **the AI only understands the question — code does the math.** The AI never calculates anything on its own. It picks from a menu of pre-approved calculations, code runs the SQL, and the result is always traceable.

---

## How To Run

### 1. Install dependencies

```
pip install google-genai pandas
```

### 2. Set up the API key

The Gemini API key is already configured in the `.env` file. If you need to update it, edit `.env`:

```
GEMINI_API_KEY=your-key-here
```

### 3. Run the assistant

```
python ai_analytics_assistant.py
```

It will first run 5 demo questions automatically, then drop into interactive mode where you can type your own questions.

---

## Dataset

**Source:** [Kaggle — Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting)

- 76,000 rows of daily retail sales data
- Columns used: Date, Region, Category, Units Sold, Inventory Level, Promotion (0/1), Seasonality
- Regions: North, South, East, West
- Categories: Electronics, Clothing, Groceries, Toys, Furniture

---

## How It Works (4 Stages)

| Stage | What Happens | Who Does It |
|-------|-------------|-------------|
| 1. Understand | Reads the question, figures out what the user wants, extracts filters (region, category, season) | AI |
| 2. Pick Metric | Selects the right calculation from the menu and plugs in the filters | Code |
| 3. Calculate | Runs SQL against the dataset, gets real numbers | Database |
| 4. Answer | Formats the numbers into a clean response + adds a source line | AI |

If the AI is not sure what the user is asking, it **asks a clarifying question** instead of guessing. A wrong answer is worse than an extra question.

---

## Metrics Menu

| Metric | What It Answers | How It Works |
|--------|----------------|-------------|
| `promo_lift` | "Did the promotion work?" | Compares avg units sold during promo vs without promo |
| `regional_volume` | "Which region sells the most?" | Sums total units sold, grouped by region |
| `inventory_delta` | "What happened to stock during promos?" | Compares avg inventory during promo vs without |
| `product_impact` | "How did a specific category do?" | Category-level sales lift during promotions |

---

## Try These Prompts

### Promotion Performance
```
Did promotions improve sales in the West region?
How did summer promotions impact units sold in the East?
```

### Regional Comparison
```
Which region had the highest sales volume during Spring?
Show me the total units sold by region
```

### Inventory Movement
```
What was the impact on inventory levels during promotions in the North?
How did inventory levels change during Winter promotions?
```

### Product Impact
```
How did Groceries perform during promotions?
Did Electronics see an improvement during promotions?
```

### Vague Questions (triggers clarification)
```
How's the business doing?
What is the weather today?
```

---

## Example Output

```
Question: Did promotions improve sales in the South region?

[Stage 1 - Understand]
  Metric: promo_lift | Filter: region = South

[Stage 2 - Pick Metric]
  Selected: promo_lift
  Compare avg units sold DURING promotions vs WITHOUT

[Stage 3 - Calculate]
  avg_during_promo = 106.75
  avg_without_promo = 83.56
  lift = +27.75%
  data_points = 15,200

[Stage 4 - Answer]
  Promotions improved sales in the South region by 27.75%,
  increasing the average from 83.56 to 106.75 across 15,200 data points.

  Source: promo_lift | Region: South | Data points: 15,200
```

---

## Tech Stack

- Python
- SQLite
- Gemini API
- Pandas
