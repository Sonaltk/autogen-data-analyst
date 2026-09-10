import asyncio
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock

async def main():
    executor = DockerCommandLineCodeExecutor(
        image="autogen-data-analyst:latest",
        work_dir="./coding_workspace",
    )

    await executor.start()

    code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)
print("Sum of column a:", df['a'].sum())
"""

    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code=code)],
        cancellation_token=CancellationToken(),
    )

    print("--- Execution Output ---")
    print(result.output)

    await executor.stop()

asyncio.run(main())
