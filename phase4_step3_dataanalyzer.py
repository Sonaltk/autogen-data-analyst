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

    data_analyzer = AssistantAgent(
        name="DataAnalyzer",
        model_client=model_client,
        system_message="""You are DataAnalyzer.
STRICT RULES:
- Your FIRST message must contain ONLY a python code block, nothing else.
- Load featured_data.csv with pandas into df.
- Compute and print (use .to_string() on any dataframe/series output so nothing truncates):
  1. Average salary by department
  2. Headcount by experience_level
  3. The single highest-paid employee (name and salary)
  4. Average years_experience by salary_band
- Do NOT say TERMINATE in this first message. Wait for real output.
- In your SECOND message: summarize the ACTUAL numbers from the executor's output only. Do not invent or recall numbers. Then say TERMINATE.
- Never say TERMINATE in the same message as code.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [data_analyzer, executor_agent],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Analyze featured_data.csv and report key insights"))

    await code_executor.stop()

asyncio.run(main())
