# AutoGen Data Analyst

An automated data analysis tool: upload a CSV, get it cleaned and analyzed, view a chart and a report, and ask follow-up questions in plain English — including requests for new charts.

Built with [AutoGen](https://microsoft.github.io/autogen/) (Microsoft's multi-agent framework), a local [Ollama](https://ollama.com/) model (free, runs entirely on-device), and a Docker sandbox for safely executing AI-generated code.

## What it does

1. **Upload a CSV** through the Streamlit web interface.
2. **Run the pipeline**: the data is loaded, deduplicated, missing values filled, a derived "band" column added, summary statistics computed, a chart generated, and a markdown report written — all executed inside an isolated Docker container.
3. **Ask questions** in plain English (e.g. "list employees in Engineering", "plot average salary by years of experience") and get an answer grounded in the real data, or a new chart generated on demand.

## Architecture

```
Streamlit frontend (app.py)
        |
        v
Docker sandbox (autogen-data-analyst:latest image)
   - pandas, matplotlib, tabulate pre-installed
   - all code execution happens here, isolated from the host machine
        |
        v
Ollama (local LLM, Qwen 2.5 7B)
   - only invoked for natural-language Q&A, not for the fixed pipeline stages
```

**Key design decision:** the six pipeline stages (load, clean, feature-engineer, analyze, visualize, report) run as **fixed, pre-written code**, not AI-generated code, executed directly in Docker. Only the natural-language Q&A feature actually invokes the LLM to write new code on the fly. This was a deliberate choice after testing showed a small local model (Qwen 2.5 7B) is unreliable at faithfully retyping/orchestrating a long sequence of tasks in one shared conversation, but is reliable at translating a single, fresh question into correct pandas code.

## Setup

Requires: Python 3.11, conda (or any virtual environment tool), Docker Desktop, Ollama.

```bash
# Create environment
conda create -n autogen-project python=3.11 -y
conda activate autogen-project

# Install dependencies
python -m pip install autogen-agentchat "autogen-ext[ollama]" "autogen-ext[docker]" pandas matplotlib streamlit tabulate

# Pull the local model
ollama pull qwen2.5:7b

# Build the Docker sandbox image
docker build -t autogen-data-analyst:latest .

# Run the app
streamlit run app.py
```

## Project structure

```
autogen-data-analyst/
├── app.py                  # Streamlit frontend (the main application)
├── Dockerfile               # Sandbox image: python:3.11-slim + pandas, matplotlib, tabulate
├── sample_data.csv          # Example dataset for testing
├── coding_workspace/        # Docker's working directory; all uploads and generated files land here
│   ├── cleaned_data.csv
│   ├── featured_data.csv
│   ├── chart.png
│   ├── report.md
│   └── qa_chart_*.png        # Charts generated from natural-language questions
├── phase3_dataloader.py     # Early prototype scripts, kept for reference
├── phase4_step*.py
├── phase5_full_pipeline.py
├── phase6_query.py
└── README.md
```

## Notable failure modes discovered during development (and how they were fixed)

Small local models are genuinely useful but have specific, repeatable failure patterns worth knowing about if you extend this project:

1. **Fabricating results without running code.** Early versions let one agent write code, summarize, and terminate all in a single message — sometimes it hallucinated a plausible-looking result instead of waiting for real execution. Fixed by splitting "write code" and "confirm result" into separate agents with a fixed turn order, so real execution is structurally guaranteed between them.
2. **Anchoring on recent chat patterns instead of following instructions.** In a long, shared multi-agent conversation, the model tended to imitate the most recently seen code pattern rather than its own distinct task. Fixed by running each pipeline stage as an isolated step (or, for fixed/deterministic stages, skipping the LLM entirely and executing pre-written code directly).
3. **Confidently describing chart images it never saw.** The Q&A confirmation agent is text-only and has no access to rendered images, but would sometimes narrate invented chart contents. Fixed by requiring the code-writing agent to print the real plotted data as text, and forbidding the confirming agent from describing anything beyond those literal printed values.
4. **Case-sensitivity mismatches.** A query like "high salary band" could silently return zero results if the underlying data used different capitalization ("High" vs "high"). Fixed by instructing all text-column comparisons to be case-insensitive.

The general lesson: **verify agent output against the real underlying files/data rather than trusting an agent's own narrated summary of what it did.**

## Known limitations

- Uses a small (7B parameter) local model for cost/privacy reasons — a larger model (e.g. GPT-4o, Claude) would likely be more reliable for complex or highly ambiguous questions, at the cost of needing an API key and internet access.
- The pipeline's feature-engineering and visualization logic is generalized but simple (e.g. always bands the first numeric column); highly irregular datasets may need custom logic.
- No authentication or multi-user support — intended for local, single-user use.