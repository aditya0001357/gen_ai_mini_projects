from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')

class TopicSummary(BaseModel):
    name: str
    summary: str = Field(description = 'The topic summary and it should be less than 200 words.')
    bibliography: str = Field(default='', description='The source of the information on  mentioned topic.')

pd_parser = PydanticOutputParser(pydantic_object=TopicSummary)

prompt = ChatPromptTemplate([
    ('system', 'Your are an expert on the topic of {topic}.\n{format_instruction}'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}'),
]).partial(format_instruction=pd_parser.get_format_instructions())

print('\nprompt = ', prompt)

chain = prompt | model | pd_parser
chat_history = []

while True:
    topic_name = input('Give the topic name or type "exit" to leave: ')
    if topic_name == 'exit':
        break
    topic_query = input('Query on the topic : ')
    result = chain.invoke({
        'topic': topic_name,
        'query': topic_query,
        'chat_history': chat_history
    })

    print(f'AI : {result.model_dump()}\n')
    chat_history.extend([
        HumanMessage(content=f'Topic: {topic_name}.\nQuery: {topic_query}'),
        AIMessage(content=result.model_dump_json())
    ])
