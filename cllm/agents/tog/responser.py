import openai
import logging

from cllm.services.nlp.llms.chat_models import ChatOpenAI
from cllm.services.nlp.llms.memory import MessageMemory
from langchain.schema import SystemMessage

from cllm.agents.base import DataType
from cllm.agents import container


RESPONSE_GENERATION_PROMPT = """Your name is ControlLLM, an AI-powered assistant developed by OpenGV-lab from Shanghai Artificial Intelligence Laboratory. For user's request, the system executes the solution and collects the results based on the following workflow. You need to respond to user requests based on the following information. 
Here are the information for you reference.

## User Request
{}

## Workflow and Execution Results
{}

Now you should pay attention to Collected Results. You first must answer the user’s request in a straightforward manner. Then you need to summarize the workflow and intermediate results friendly. Some of the results may not be accurate and need you to use your judgement in making decisions. If the results contain file names, you have to output the file name directly. Only if there is nothing returned by tools, you should tell user you can not finish the task. Now, please friendly summarize the results and answer the question for the user requests `{}`.
""".strip()


SIMPLE_RESPONSE_GENERATION_PROMPT = """Your name is ControlLLM, an AI-powered assistant developed by OpenGVLab from Shanghai Artificial Intelligence Laboratory. You need to respond to user requests based on the following information.
Here are the information for you reference.

## User Request
{}

## Workflow and Execution Results
{}

Now, please friendly summarize the results and answer the question for the user requests `{}`.
""".strip()

logger = logging.getLogger(__name__)


def generate_response(user_input, solution, output_files):
    if (
        len(solution) <= 1
        and len(solution[0]) <= 1
        and solution[0][0].tool_name == "question_answering"
    ):
        content = SIMPLE_RESPONSE_GENERATION_PROMPT.format(
            user_input, solution, user_input
        )
    else:
        content = RESPONSE_GENERATION_PROMPT.format(user_input, solution, user_input)

    logger.info("##### Response Generation #####")
    logger.info(content)

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    messages = [SystemMessage(content=content)]
    output = chat(messages)
    logger.info(output)

    # files = [output for output in output_files if isinstance(output, container.File)]
    # return [container.Text('Response', DataType.TEXT, output)] + files
    return [container.Text("Response", DataType.TEXT, output)]
