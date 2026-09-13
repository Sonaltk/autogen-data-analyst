import streamlit as st
import asyncio
import os
import time
import pandas as pd
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock

WORK_DIR = "./coding_workspace"
os.makedirs(WORK_DIR, exist_ok=True)


def get_stages(csv_filename):
    return {
        "DataLoader": f"""
import pandas as pd
df = pd.read_csv('{csv_filename}')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head().to_string())
""",
        "DataCleaner": f"""
import pandas as pd
df = pd.read_csv('{csv_filename}')
rows_before = len(df)
df.drop_duplicates(inplace=True)
numeric_cols = df.select_dtypes(include=['number']).columns
text_cols = df.select_dtypes(exclude=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[text_cols] = df[text_cols].fillna('Unknown')
rows_after = len(df)
df.to_csv('cleaned_data.csv', index=False)
print(f"Rows before: {{rows_before}}, Rows after: {{rows_after}}")
print(df.to_string())
""",
        "FeatureEngineer": """
import pandas as pd
df = pd.read_csv('cleaned_data.csv')
if 'salary' in df.columns:
    df['salary_band'] = pd.qcut(df['salary'], 3, labels=['Low', 'Medium', 'High'])
df.to_csv('featured_data.csv', index=False)
print(df.to_string())
""",
        "DataAnalyzer": """
import pandas as pd
df = pd.read_csv('featured_data.csv')
print(df.describe(include='all').to_string())
""",
        "Visualizer": """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
df = pd.read_csv('featured_data.csv')
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    cat_cols = df.select_dtypes(exclude=['number']).columns
    if len(cat_cols) > 0:
        group_col = cat_cols[0]
        avg = df.groupby(group_col)[col].mean().reset_index()
        plt.figure(figsize=(10, 6))
        plt.bar(avg[group_col].astype(str), avg[col])
        plt.title(f'Average {col} by {group_col}')
        plt.xlabel(group_col)
        plt.ylabel(f'Average {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('chart.png')
        plt.close()
        print("File exists:", os.path.exists('chart.png'))
    else:
        plt.figure(figsize=(10, 6))
        df[col].hist()
        plt.title(f'Distribution of {col}')
        plt.savefig('chart.png')
        plt.close()
        print("File exists:", os.path.exists('chart.png'))
else:
    print("No numeric columns found to chart.")
""",
        "ReportWriter": """
import pandas as pd
import os
df = pd.read_csv('featured_data.csv')
summary_md = df.describe(include='all').to_markdown()
report_content = f'''# Data Analysis Report

This report summarizes the uploaded dataset after cleaning and feature engineering.

## Summary Statistics

{summary_md}

See `chart.png` for a visual breakdown.
'''
with open('report.md', 'w') as f:
    f.write(report_content)
print(open('report.md').read())
print("Report file exists:", os.path.exists('report.md'))
""",
    }


async def run_pipeline(csv_filename, status_placeholder):
    code_executor = DockerCommandLineCodeExecutor(image="autogen-data-analyst:latest", work_dir=WORK_DIR)
    await code_executor.start()
    stages = get_stages(csv_filename)
    logs = {}
    try:
        for stage_name, code in stages.items():
            status_placeholder.write(f"Running: {stage_name}...")
            result = await code_executor.execute_code_blocks(
                code_blocks=[CodeBlock(language="python", code=code)],
                cancellation_token=CancellationToken(),
            )
            logs[stage_name] = result.output
            if result.exit_code != 0:
                status_placeholder.error(f"Stage {stage_name} failed. See logs below.")
                break
    finally:
        await code_executor.stop()
    return logs


async def ask_question(user_question, unique_id):
    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")
    code_executor = DockerCommandLineCodeExecutor(image="autogen-data-analyst:latest", work_dir=WORK_DIR)
    await code_executor.start()

    executor_agent = CodeExecutorAgent(name="code_executor", code_executor=code_executor)

    try:
        df_preview = pd.read_csv(os.path.join(WORK_DIR, "featured_data.csv"))
        columns_list = ", ".join(df_preview.columns.tolist())
        text_cols = df_preview.select_dtypes(exclude=['number']).columns.tolist()
        sample_values = {c: df_preview[c].dropna().unique()[:5].tolist() for c in text_cols}
    except Exception:
        columns_list = "unknown"
        sample_values = {}

    chart_filename = f"qa_chart_{unique_id}.png"

    query_analyst = AssistantAgent(
        name="QueryAnalyst",
        model_client=model_client,
        system_message=f"""You are QueryAnalyst, a data analysis code writer.
The file 'featured_data.csv' has these exact columns: {columns_list}
Sample real values for text columns: {sample_values}
The user's question is: "{user_question}"

STRICT RULES:
- Your ONLY message must contain ONE python code block. It MUST start with three backticks followed immediately by the word python:
```python
(your code here)
```
- No commentary, no text before or after the code block.
- Load featured_data.csv with pandas as df.
- Read the question carefully and identify EXACTLY which column(s) it refers to. Do not substitute a different column than the one named or implied in the question.
- When filtering text/categorical columns, ALWAYS compare case-insensitively.
- If the question asks to "show", "plot", "chart", "visualize", "graph", or otherwise implies a visual:
  - Start with: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import os
  - Build the chart using the CORRECT column(s) from the actual question.
  - Save it using EXACTLY this filename: plt.savefig('{chart_filename}')
  - Then plt.close()
  - CRITICAL: after saving, print the exact data table/values that were plotted (e.g. print(the_grouped_dataframe.to_string())), so the underlying numbers are visible as real text, not just in the image.
  - Also print os.path.exists('{chart_filename}') to confirm.
- If the question does NOT ask for a visual, just print text/table results using .to_string() so nothing truncates, and print the row count if returning a list. Do not create a chart in this case.
- If ambiguous, pick a reasonable interpretation and print it explicitly first.
- Do not say TERMINATE.""",
    )

    answer_confirmer = AssistantAgent(
        name="AnswerConfirmer",
        model_client=model_client,
        system_message=f"""You are AnswerConfirmer.
The user asked: "{user_question}"

CRITICAL RULES:
- You CANNOT see any image. You only see printed text from code_executor above. Never describe, guess, or narrate what a chart "shows" beyond the exact printed numbers you can see in that text.
- If the output contains an error, a traceback, or "No code blocks found", say exactly: "I couldn't compute an answer because the code failed to run. Error details: [quote the real error]." Then say TERMINATE.
- If a chart was created (file exists: True) AND real printed data values are present, say "Here's the chart you asked for, based on this data:" and then quote the REAL printed values exactly (copy them, do not reformat, invent, or round differently). Do not add any interpretation of the image itself.
- If a chart was supposedly created but NO real data values were printed to confirm it, say: "A chart file was created, but I can't verify its contents were correct without printed data alongside it." Do not guess what it contains.
- If it's a text/table answer (no chart), answer in plain English using ONLY values that actually appear above. Never invent names, numbers, or rows.
- If the result has more than 3 items, list all of them and state the total count.
Then say TERMINATE.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(4)
    team = RoundRobinGroupChat([query_analyst, executor_agent, answer_confirmer], termination_condition=termination)

    messages = []
    async for msg in team.run_stream(task=user_question):
        if hasattr(msg, "source") and hasattr(msg, "content"):
            messages.append((msg.source, msg.content))

    await code_executor.stop()

    chart_path = os.path.join(WORK_DIR, chart_filename)
    chart_result = chart_path if os.path.exists(chart_path) else None

    return messages, chart_result


st.set_page_config(page_title="AutoGen Data Analyst", layout="wide")
st.title("Automated Data Analyst")
st.caption("Powered by AutoGen + local Ollama (Qwen 2.5) + Docker sandbox")

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    save_path = os.path.join(WORK_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.csv_filename = uploaded_file.name
    st.success(f"Saved: {uploaded_file.name}")
    st.dataframe(pd.read_csv(save_path).head())

    if st.button("Run Full Analysis Pipeline"):
        status = st.empty()
        with st.spinner("Running pipeline..."):
            logs = asyncio.run(run_pipeline(uploaded_file.name, status))
        status.write("Pipeline finished.")
        st.session_state.pipeline_done = True
        with st.expander("Show raw pipeline logs"):
            for stage, output in logs.items():
                st.text(f"--- {stage} ---\n{output}")

if st.session_state.pipeline_done:
    st.header("Results")
    col1, col2 = st.columns(2)
    with col1:
        featured_path = os.path.join(WORK_DIR, "featured_data.csv")
        if os.path.exists(featured_path):
            st.subheader("Processed Data")
            st.dataframe(pd.read_csv(featured_path))
    with col2:
        chart_path = os.path.join(WORK_DIR, "chart.png")
        st.subheader("Chart")
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            st.warning(f"No chart found at {chart_path}")

    report_path = os.path.join(WORK_DIR, "report.md")
    if os.path.exists(report_path):
        st.subheader("Report")
        with open(report_path) as f:
            st.markdown(f.read())

    st.header("Ask a question about your data")
    st.caption("Tip: ask things like 'plot average salary by department' to get a chart, or 'list employees in Sales' for a text answer.")
    user_question = st.text_input("Your question")
    if st.button("Ask") and user_question:
        with st.spinner("Thinking..."):
            unique_id = str(int(time.time() * 1000))
            messages, chart_result = asyncio.run(ask_question(user_question, unique_id))
        st.session_state.qa_history.append((user_question, messages, chart_result))

    for q, messages, chart_result in reversed(st.session_state.qa_history):
        st.markdown(f"**Q: {q}**")
        for source, content in messages:
            if source == "AnswerConfirmer":
                st.markdown(f"**Answer:** {content}")
        if chart_result:
            st.image(chart_result, use_container_width=True)
        with st.expander("Show reasoning"):
            for source, content in messages:
                st.text(f"[{source}]\n{content}")