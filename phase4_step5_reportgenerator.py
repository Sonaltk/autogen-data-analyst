import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async def main():
    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")

    code_executor = DockerCommandLineCodeExecutor(
        image="autogen-data-analyst:latest",
        work_dir="./coding_workspace",
    )
    await code_executor.start()

    # Two separate agent wrappers around the SAME underlying executor,
    # so we can place them at fixed, distinct points in a strict turn order.
    executor_1 = CodeExecutorAgent(name="code_executor_1", code_executor=code_executor)
    executor_2 = CodeExecutorAgent(name="code_executor_2", code_executor=code_executor)

    report_writer = AssistantAgent(
        name="ReportWriter",
        model_client=model_client,
        system_message="""You are ReportWriter. Your ONLY job is to output ONE python code block.
Copy this code EXACTLY, with nothing before or after it:

```python
import pandas as pd
import os

df = pd.read_csv('featured_data.csv')

avg_salary_by_dept = df.groupby('department')['salary'].mean().reset_index()
headcount_by_exp_level = df['experience_level'].value_counts().reset_index()
headcount_by_exp_level.columns = ['experience_level', 'count']
highest_paid_employee = df.loc[[df['salary'].idxmax()]][['name', 'department', 'salary', 'years_experience', 'salary_band', 'experience_level']]

avg_salary_md = avg_salary_by_dept.to_markdown(index=False)
headcount_md = headcount_by_exp_level.to_markdown(index=False)
highest_paid_md = highest_paid_employee.to_markdown(index=False)

report_content = f'''# Salary Analysis Report

This report summarizes salary data across departments and experience levels.

## Average Salary by Department

{avg_salary_md}

## Headcount by Experience Level

{headcount_md}

## Highest-Paid Employee

{highest_paid_md}

See `salary_by_department.png` for a visual breakdown.
'''

with open('report.md', 'w') as f:
    f.write(report_content)

print(open('report.md').read())
print("File exists:", os.path.exists('report.md'))
```

You are not permitted to write anything else, ever. You do not summarize results. You do not confirm anything. That is someone else's job.""",
    )

    report_confirmer = AssistantAgent(
        name="ReportConfirmer",
        model_client=model_client,
        system_message="""You are ReportConfirmer. Look at the REAL output printed by code_executor_1 above.
Quote a short real snippet of it, confirm "File exists: True" appeared, then say TERMINATE.
Only use text that actually appears in that real output. Never invent content.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [report_writer, executor_1, report_confirmer, executor_2],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Generate a markdown report from featured_data.csv"))

    await code_executor.stop()

asyncio.run(main())
