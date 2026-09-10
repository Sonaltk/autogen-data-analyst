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

    data_loader = AssistantAgent(
        name="DataLoader",
        model_client=model_client,
        system_message="""You are DataLoader, a data loading specialist.

STRICT RULES:
- Your FIRST message must contain ONLY a python code block. No commentary, no summary, no TERMINATE.
- The code should load the CSV with pandas, then print: shape, column names, and the first 5 rows.
- Do NOT write example output, placeholder values, or guess what the data looks like.
- Wait for code_executor to give you the REAL output in the next message.
- Only in your SECOND message: read the actual output from code_executor, summarize it in plain English using the real numbers you see, then say TERMINATE.
- Never say TERMINATE in the same message as your code.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [data_loader, executor_agent],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Load and inspect the CSV file at sample_data.csv"))

    await code_executor.stop()

asyncio.run(main())
