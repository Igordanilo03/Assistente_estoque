import os
import streamlit as st
from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer


os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')

st.set_page_config(
    page_title='Assistente GPT',
    page_icon='📝'
    )

st.header('Assistente pessoal 🤖')

model_option = [
    'llama3-70b-8192', 
    'mixtral-8x7b-32768',
    'llama-3.3-70b-versatile',
    'deepseek-r1-distill-llama-70b',
    'llama-3.2-90b-vision-preview',
]

sected_model = st.sidebar.selectbox(
    label='Selecione seu modelo.',
    options=model_option
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Esse agente busca informações na internet, e responde o usuário com base na pergunta.')

st.write('Faça sua pergunta')
user_question = st.text_input('O que deseja saber?')

model = ChatGroq(
    model=sected_model,
)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python Repl',
    description='Um shell python. Use isso para executar código python. Execute apenas códigos python validos.'
                'Se precisar obter um retorno use a função "print(...)".'
                'Use isso para executar calculos e automações.',
    func=python_repl.run
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name='DuDuckGo Search',
    description='Faça busca na internet, e traga todas as informações que o usuario solicitar.Use para encontrar respostas com base na expecificação do usuario. Traga somente as informações da internet, não use dados no seu armazenamento',
    func=search.run
)

def web_scrape_tool(url: str):
    try:
        loader = AsyncChromiumLoader([url])
        docs = loader.load()
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=['p', 'table', 'div'])
        return docs_transformed[0].page_content
    except Exception as e:
        return f'Falha ao fazer scraping da URL {url}: {str(e)}'
    
scrape_tool = Tool(
    name='Web Scraper',
    description='Faz scraping de uma URL fornecida e retorna o conteudo {parágrafo, tabelas, etc.}',
    func=web_scrape_tool
)

react_instruction = hub.pull('hwchase17/react')
tools = [python_repl_tool, search_tool, scrape_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instruction,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

prompt = '''
Com base na pergunta do usuário, siga estes passos:
1. Use o DuckDuckGo Search para encontrar uma URL relevante.
2. Use o Web Scraper para extrair o conteúdo dessa URL.
3. Use o Python REPL para realizar cálculos ou automações com os dados extraídos, se aplicável.
4. Retorne uma resposta clara e amigável em português brasileiro.
Pergunta: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

if st.button('Buscar'):
    with st.spinner('Buscando informações...'):
        if user_question:
            formatted_question = prompt_template.format(q=user_question)
            output = agent_executor.invoke({'input': formatted_question})
            st.markdown(output.get('output'))
        
        else:
            st.warning('Porfavor faça uma pergunta')
        