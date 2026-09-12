import asyncio
import os
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

STAGES = {
    "DataLoader": """
import pandas as pd
df = pd.read_csv('sample_data.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head().to_string())
""",
    "DataCleaner": """
import pandas as pd
df = pd.read_csv('sample_data.csv')
rows_before = len(df)
df.drop_duplicates(inplace=True)
numeric_cols = df.select_dtypes(include=['number']).columns
text_cols = df.select_dtypes(exclude=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[text_cols] = df[text_cols].fillna('Unknown')
rows_after = len(df)
df.to_csv('cleaned_data.csv', index=False)
print(f"Rows before: {rows_before}, Rows after: {rows_after}")
print(df.to_string())
""",
    "FeatureEngineer": """
import pandas as pd
df = pd.read_csv('cleaned_data.csv')
df['salary_band'] = df['salary'].apply(lambda x: 'Low' if x < 70000 else ('Medium' if x < 100000 else 'High'))
df['experience_level'] = df['years_experience'].apply(lambda x: 'Junior' if x < 3 else ('Mid' if x < 6 else 'Senior'))
df.to_csv('featured_data.csv', index=False)
print(df.to_string())
""",
    "DataAnalyzer": """
import pandas as pd
df = pd.read_csv('featured_data.csv')
print("Average salary by department:")
print(df.groupby('department')['salary'].mean().reset_index().to_string())
print("\\nHeadcount by experience level:")
print(df['experience_level'].value_counts().to_string())
print("\\nHighest paid employee:")
print(df.loc[[df['salary'].idxmax()]].to_string())
""",
    "Visualizer": """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
df = pd.read_csv('featured_data.csv')
avg = df.groupby('department')['salary'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(avg['department'], avg['salary'])
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.savefig('salary_by_department.png')
plt.close()
print("File exists:", os.path.exists('salary_by_department.png'))
print("File size:", os.path.getsize('salary_by_department.png'))
""",
    "ReportWriter": """
import pandas as pd
import os
df = pd.read_csv('featured_data.csv')
avg_salary_md = df.groupby('department')['salary'].mean().reset_index().to_markdown(index=False)
headcount = df['experience_level'].value_counts().reset_index()
headcount.columns = ['experience_level', 'count']
headcount_md = headcount.to_markdown(index=False)
highest_paid = df.loc[[df['salary'].idxmax()]][['name', 'department', 'salary', 'years_experience', 'salary_band', 'experience_level']]
highest_paid_md = highest_paid.to_markdown(index=False)
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
print("Report file exists:", os.path.exists('report.md'))
""",
}


async def main():
    code_executor = DockerCommandLineCodeExecutor(
        image="autogen-data-analyst:latest",
        work_dir="./coding_workspace",
    )
    await code_executor.start()

    for stage_name, code in STAGES.items():
        print(f"\n{'='*20} STAGE: {stage_name} {'='*20}")
        result = await code_executor.execute_code_blocks(
            code_blocks=[CodeBlock(language="python", code=code)],
            cancellation_token=CancellationToken(),
        )
        print(result.output)
        if result.exit_code != 0:
            print(f"!!! STAGE {stage_name} FAILED with exit code {result.exit_code} !!!")
            break

    await code_executor.stop()

    print(f"\n{'='*20} FINAL VERIFICATION (plain Python, no AI) {'='*20}")
    for fname in ["cleaned_data.csv", "featured_data.csv", "salary_by_department.png", "report.md"]:
        path = os.path.join("coding_workspace", fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"{fname}: exists={exists}, size={size} bytes")

asyncio.run(main())
