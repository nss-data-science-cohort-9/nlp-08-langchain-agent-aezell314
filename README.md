## Bonus Exercise: Build a Chat-with-Data-Agent

**Goal:** The goal of this exercise is to build a simple LLM-powered data analyst agent that can answer questions about the Titanic dataset.

Instead of writing pandas code directly, you'll:
* expose a small set of safe Python tools
* connect to an LLM using LangChang
* allow the model to decide which analysis to run

This is a simplified version of real-world "data copilot" systems.

#### Dataset

You'll use the Titanic dataset, which can be loaded from seaborn:

```
import seaborn as sns
titanic = sns.load_dataset("titanic")
```

The other imports you'll need are as follows:

```
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

import json
```


### Part 1: Explore the Dataset (No LLM yet)

Before building anything, write code to print column names, the shape of the dataset, and the first 5 rows.

Then, write pandas code to find the average survival rate and the survival rate grouped by sex.

### Part 2: Build Safe Data Tools

Now, create functions that an LLM can be allowed to use. **Warning:** These functions should not use `eval`, `exec`, or raw Python execution. We don't want the language model executing arbitrary Python code!

When doing this, we'll wrap our functions in the `@tool` [decorator](https://www.w3schools.com/PYTHON/python_decorators.asp). See https://docs.langchain.com/oss/python/langchain/tools 

For example, we a safe function to get the column names would be:

```
@tool
def get_columns() -> str:
    """Return dataset column names."""
    return str(list(df.columns))
```

Build other functions that may be helpful. For example, fill in the code for the following functions:

```
@tool
def group_mean(group_col, target_col):
    ...
```
This should group the dataset, compute the mean of target_col, and return the result as a string. Hint: You can use the to_string() method.

```
@tool
def count_where(column, value):
    ...
```
This function should return how many rows match a condition.

After defining tools, create a tools list:

```
tools = [get_columns, group_mean, count_where]
```

Be sure to include any other tools that you defined.

### Part 3: Build the Prompt

Now, we need to build a prompt. We'll use the [ChatPromptTemplate from_messages function](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate/from_messages).

```
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst assistant.\n"
     "Use tools when needed to answer questions about the dataset.\n"
     "Be concise and explain results clearly."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
```
Notice that we're giving it an agent_scratchpad, which serves as a "working memory" for the agent's intermediate steps. It contains things like which tool the model decided to use, the input and output, and any intermediate reasoning steps.

### Part 4: Connect to an LLM

Create a ChatOpenAI instance, connecting to openrouter using the owl-alpha model.

### Part 5:  Build the Agent

Using [the create_tool_calling_agent function](https://reference.langchain.com/python/langchain-classic/agents/tool_calling_agent/base/create_tool_calling_agent), combine the ChatOpenAI instance, the tools, and the prompt into an agent.

Finally, create an [AgentExecutor](https://reference.langchain.com/python/langchain-classic/agents/agent/AgentExecutor). 

```
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

Notice that we have verbose set to True so we can see the agent's reasoning.

### Part 6: Ask Questions

Test your agent. Ask questions like:
* Did women survive more often than men?
* Which passenger class had the highest survival rate?
* How many first-class passengers were there?

You may find that you need to provide additional tools or to tweak the system prompt if the agent makes mistakes.