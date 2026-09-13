import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

USER_QUESTION = "Which department has the most employees, and what's their average years of experience?"

async def main():
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

The user's question is: "{USER_QUESTION}"

STRICT RULES:
- Your ONLY message must contain ONE python code block, nothing else, no commentary.
- Write pandas code that loads featured_data.csv and computes whatever is needed to answer the question.
- Print the result clearly using .to_string() on any dataframe/series so nothing truncates.
- Do not say TERMINATE. Do not add any text before or after the code block.""",
    )

    answer_confirmer = AssistantAgent(
        name="AnswerConfirmer",
        model_client=model_client,
        system_message=f"""You are AnswerConfirmer.
The user asked: "{USER_QUESTION}"
Look at the REAL output from code_executor above. Answer the user's question in plain English, using ONLY numbers/values that actually appear in that real output. Do not invent anything.
Then say TERMINATE.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(4)

    team = RoundRobinGroupChat(
        [query_analyst, executor_agent, answer_confirmer],
        termination_condition=termination,
    )

    await Console(team.run_stream(task=USER_QUESTION))

    await code_executor.stop()

asyncio.run(main())
