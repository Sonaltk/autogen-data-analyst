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

    feature_engineer = AssistantAgent(
        name="FeatureEngineer",
        model_client=model_client,
        system_message="""You are FeatureEngineer.
STRICT RULES:
- Your FIRST message must contain ONLY a python code block, nothing else.
- Load cleaned_data.csv with pandas into df.
- Create 'salary_band': 'Low' if salary < 70000, 'Medium' if 70000 <= salary < 100000, 'High' if salary >= 100000.
- Create 'experience_level': 'Junior' if years_experience < 3, 'Mid' if 3 <= years_experience < 6, 'Senior' if years_experience >= 6.
- Save using df.to_csv('featured_data.csv', index=False).
- IMPORTANT: print ONLY these columns so nothing gets truncated:
  print(df[['name', 'salary', 'salary_band', 'years_experience', 'experience_level']].to_string())
- Do NOT say TERMINATE in this first message. Wait for real output.
- In your SECOND message: you may ONLY use numbers you can literally see in the executor's output above. Do not invent, estimate, or recall numbers from memory. Quote 2-3 real rows exactly as printed, confirm the labels match the rules. If correct, say TERMINATE.
- Never say TERMINATE in the same message as code.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [feature_engineer, executor_agent],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Add salary_band and experience_level columns to cleaned_data.csv"))

    await code_executor.stop()

asyncio.run(main())
