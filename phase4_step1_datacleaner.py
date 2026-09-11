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

    executor_agent = CodeExecutorAgent(
        name="code_executor",
        code_executor=code_executor,
    )

    data_cleaner = AssistantAgent(
        name="DataCleaner",
        model_client=model_client,
        system_message="""You are DataCleaner.
STRICT RULES:
- Your FIRST message must contain ONLY a python code block, nothing else.
- Load sample_data.csv with pandas into df.
- Check for and PRINT: missing values per column, number of duplicate rows.
- Then clean: drop duplicate rows, fill missing numeric values with the column median, fill missing text values with 'Unknown'.
- Save the cleaned data using df.to_csv('cleaned_data.csv', index=False).
- Print confirmation showing row counts before and after cleaning.
- Wait for real output before summarizing.
- In your SECOND message, summarize the real cleaning results using actual numbers you saw, then say TERMINATE.
- Never say TERMINATE in the same message as code.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [data_cleaner, executor_agent],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Load, inspect, and clean the CSV file at sample_data.csv"))

    await code_executor.stop()

asyncio.run(main())
