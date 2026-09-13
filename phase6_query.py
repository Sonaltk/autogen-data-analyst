import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async def main():
    user_question = input("Ask a question about the employee data: ").strip()
    if not user_question:
        user_question = "What is the average salary overall?"

    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")

    code_executor = DockerCommandLineCodeExecutor(
        image="autogen-data-analyst:latest",
        work_dir="./coding_workspace",
    )
    await code_executor.start()

    executor_agent = CodeExecutorAgent(name="code_executor", code_executor=code_executor)

    query_analyst = AssistantAgent(
        name="QueryAnalyst",
        model_client=model_client,
        system_message=f"""You are QueryAnalyst, a data analysis code writer.

The file 'featured_data.csv' has these exact columns (lowercase): name, department, salary, years_experience, salary_band, experience_level.
Possible values: salary_band is one of Low/Medium/High. experience_level is one of Junior/Mid/Senior.

The user's question is: "{user_question}"

STRICT RULES:
- Your ONLY message must contain ONE python code block. It MUST start with three backticks followed immediately by the word python, like this exact format:
```python
(your code here)
```
- No commentary, no text before or after the code block. The code block markers are mandatory or the system cannot run your code at all.
- Write pandas code that loads featured_data.csv and computes whatever is needed to answer the question.
- If ambiguous, pick a reasonable interpretation and print it explicitly first.
- Print results with .to_string() so nothing truncates, and print the row count.
- Do not say TERMINATE.""",
    )

    answer_confirmer = AssistantAgent(
        name="AnswerConfirmer",
        model_client=model_client,
        system_message=f"""You are AnswerConfirmer.
The user asked: "{user_question}"

CRITICAL SAFETY RULE: Look at the REAL output from code_executor above.
- If it contains an error, a traceback, or a message like "No code blocks found", DO NOT attempt to answer the question. Instead say exactly: "I couldn't compute an answer because the code failed to run. Error details: [quote the real error]." Then say TERMINATE.
- If it contains real, valid data, answer the question in plain English using ONLY values that actually appear in that real output. Never invent names, numbers, or rows that are not literally present above.
- If the result is a list of more than 3 items, list all of them and state the total count.
Then say TERMINATE.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(4)

    team = RoundRobinGroupChat(
        [query_analyst, executor_agent, answer_confirmer],
        termination_condition=termination,
    )

    await Console(team.run_stream(task=user_question))

    await code_executor.stop()

asyncio.run(main())
