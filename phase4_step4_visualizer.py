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

    visualizer = AssistantAgent(
        name="Visualizer",
        model_client=model_client,
        system_message="""You are Visualizer.
STRICT RULES:
- Your FIRST message must contain ONLY a python code block, nothing else.
- Start the code with EXACTLY these two lines, in this order:
  import matplotlib
  matplotlib.use('Agg')
  Then import matplotlib.pyplot as plt AFTER those two lines.
- Load featured_data.csv with pandas into df. The real column names (lowercase) are:
  name, department, salary, years_experience, salary_band, experience_level
- Create a bar chart of average salary by department (columns: 'department', 'salary'). Add a title and axis labels.
- Save it as salary_by_department.png using plt.savefig(), then plt.close().
- After saving, print the exact filesystem path, then use os.path.exists() and os.path.getsize() to confirm the file was actually created and print the results.
- Do NOT say TERMINATE in this first message. Wait for real output.
- In your SECOND message: confirm ONLY based on the executor's real output that the file exists and has a nonzero size. Then say TERMINATE.
- Never say TERMINATE in the same message as code.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [visualizer, executor_agent],
        termination_condition=termination,
    )

    await Console(team.run_stream(task="Create a bar chart of average salary by department from featured_data.csv"))

    await code_executor.stop()

asyncio.run(main())
