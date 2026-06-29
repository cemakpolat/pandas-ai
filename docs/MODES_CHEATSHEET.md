╔══════════════════════════════════════════════════════════════════════════════╗
║                 PyChartAI Execution Modes: Quick Reference                  ║
╚══════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣  DEFAULT MODE (Recommended - Use This!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Installation:     pip install pychartai
  
  Code:             sdf.chat("What is revenue by region?")
  
  How it works:     LLM → generates Python → RestrictedSandbox executes
  
  LLM calls:        1 (Fast!)
  
  pandasai:         ❌ NOT required
  
  Security:         ✅ Sandboxed (RestrictedPython blocks dangerous ops)
  
  Speed:            ⚡⚡⚡ Fastest
  
  Best for:         • Simple analytics ("top 5 products")
                    • Aggregations (group by, sum, avg)
                    • Most real-world queries


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣  AGENT MODE (Advanced - SQL Decomposition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Installation:     pip install pychartai[pandasai]
  
  Code:             sdf.chat("What is revenue by region?", use_agent=True)
  
  How it works:     LLM → SQL → Execute SQL → LLM → Python → Execute Python
  
  LLM calls:        2-3 (Slower, but more sophisticated)
  
  pandasai:         ✅ REQUIRED
  
  Security:         ❌ Uses exec() (less secure than sandbox)
  
  Speed:            🐌 Slower (multiple LLM calls)
  
  Best for:         • Complex SQL queries (joins, unions)
                    • Multi-step reasoning
                    • When you want pandasai orchestration


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣  CUSTOM MODE (Power Users - Build Your Own)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Installation:     pip install pychartai
  
  Code:             orchestrator = PyChartAICustomOrchestrator(llm)
                    result = orchestrator.execute(df, query)
  
  How it works:     Query classification → Type-optimized prompt → Execute
  
  LLM calls:        1 (or custom)
  
  pandasai:         ❌ NOT required
  
  Security:         ✅ Sandboxed
  
  Speed:            ⚡⚡ Fast
  
  Best for:         • Domain-specific logic
                    • Query-type aware generation
                    • Custom validation


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE GENERATION COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    │  DEFAULT   │   AGENT    │   CUSTOM
────────────────────┼────────────┼────────────┼──────────
Code Gen Strategy   │ Direct     │ SQL+Python │ Optimized
LLM Calls          │ 1          │ 2-3        │ 1
Installation       │ Basic      │ W/pandasai │ Basic
pandasai Required  │ NO         │ YES        │ NO
Sandbox Isolated   │ YES        │ NO         │ YES
Speed              │ ⚡⚡⚡      │ 🐌        │ ⚡⚡
Customizable       │ Limited    │ Moderate   │ FULL
Learning Curve     │ Easy       │ Hard       │ Medium
Security           │ High       │ Medium     │ High
Use Case Best For  │ Most       │ Complex    │ Special


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW CODE GENERATION WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT MODE:
  1. Prompt: "You have df with columns X, Y, Z. Query: What is revenue by region?"
  2. LLM responds: "df.groupby('region')['revenue'].sum()"
  3. Sandbox executes code with df pre-loaded
  4. Returns result

AGENT MODE:
  1. Prompt: "Write SQL for: revenue by region"
  2. LLM responds: "SELECT region, SUM(revenue) FROM df GROUP BY region"
  3. Agent executes SQL → intermediate result
  4. Prompt: "Using SQL result, format for visualization"
  5. LLM responds: "bar_chart(df_result, ...)"
  6. Agent executes Python
  7. Returns result

CUSTOM MODE:
  1. Detect query type: "aggregation"
  2. Build type-optimized prompt emphasizing groupby/agg
  3. LLM responds: optimized code
  4. Validate result (your logic)
  5. Sandbox executes
  6. Returns result


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK DECISION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question: What should I use?

├─ "I'm just getting started"
│  └─ Use DEFAULT mode
│     Code: sdf.chat("query")
│
├─ "I have simple analytics queries"
│  └─ Use DEFAULT mode
│     Code: sdf.chat("What is revenue by product?")
│
├─ "I need SQL decomposition for complex queries"
│  └─ Use AGENT mode
│     Code: sdf.chat("query", use_agent=True)
│
├─ "I need custom logic/validation"
│  └─ Use CUSTOM mode
│     Code: orchestrator.execute(df, query)
│
├─ "I want maximum speed"
│  └─ Use DEFAULT mode
│     (1 LLM call is faster than 2-3)
│
├─ "I want maximum security"
│  └─ Use DEFAULT or CUSTOM mode
│     (Both use RestrictedSandbox)
│
└─ "I don't want to install pandasai"
   └─ Use DEFAULT or CUSTOM mode
      (Both work without pandasai)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT MODE:
  import pychartai as pai
  
  llm = pai.OllamaLLM(model='llama3.2')
  pai.config.set({'llm': llm})
  
  df = pai.read_csv('data.csv')
  sdf = pai.SmartDataFrame(df)
  
  # Just use it!
  result = sdf.chat("What is revenue by region?")


AGENT MODE:
  import pychartai as pai
  
  llm = pai.OllamaLLM(model='llama3.2')
  pai.config.set({'llm': llm})
  
  df = pai.read_csv('data.csv')
  sdf = pai.SmartDataFrame(df)
  
  # Opt-in with use_agent=True
  result = sdf.chat("What is revenue by region?", use_agent=True)


CUSTOM MODE:
  from examples.custom_orchestrator_example import PyChartAICustomOrchestrator
  import pychartai as pai
  
  llm = pai.OllamaLLM(model='llama3.2')
  orchestrator = PyChartAICustomOrchestrator(llm, verbose=False)
  
  df = pai.read_csv('data.csv')
  
  # Use orchestrator
  result = orchestrator.execute(df, "What is revenue by region?")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUNNING EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python examples/execution_modes_examples.py --example 1  # Default mode
python examples/execution_modes_examples.py --example 2  # Agent mode
python examples/execution_modes_examples.py --example 3  # Custom mode
python examples/execution_modes_examples.py --example 4  # Comparison
python examples/execution_modes_examples.py --example 5  # Multi-turn conversation


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CODE GENERATION STILL HAPPENS in all three modes
   └─ Just different strategies (direct, orchestrated, optimized)

✅ All three modes generate & execute code
   └─ DEFAULT: df.groupby(...).sum()
   └─ AGENT: SQL then Python
   └─ CUSTOM: query-type-optimized code

✅ Your existing examples work unchanged
   └─ They automatically use DEFAULT mode

✅ pandasai is now OPTIONAL
   └─ Only needed for AGENT mode

✅ Security has improved
   └─ DEFAULT and CUSTOM use RestrictedSandbox

✅ Speed has improved
   └─ DEFAULT is 2-3x faster than AGENT

✅ Decision is simple
   └─ Start with DEFAULT
   └─ Add AGENT only if you need SQL decomposition
   └─ Custom only if you need domain-specific logic


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read these in order:

1. CODE_GENERATION_EXPLAINED.md
   └─ Your main question answered

2. EXECUTION_FLOWS.md
   └─ Visual diagrams & data flows

3. ARCHITECTURE_DEEPDIVE.md
   └─ Technical details

4. examples/execution_modes_examples.py
   └─ Working code

5. examples/custom_orchestrator_example.py
   └─ Custom agent templates


╔══════════════════════════════════════════════════════════════════════════════╗
║                         QUICK START: Use DEFAULT                           ║
║                                                                            ║
║  sdf.chat("What is revenue by region?")                                   ║
║                                                                            ║
║  That's it! It works, it's fast, it's secure.                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
