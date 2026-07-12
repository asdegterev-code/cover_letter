#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from typing import Optional, Literal, List, Any
import operator, re
from operator import add
from collections import deque
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import convert_runnable_to_tool
#from langchain_core.documents import Document
#from langchain_community.tools import DuckDuckGoSearchRun
import asyncio, json, os, time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq

CV = "Lebenslauf-etc.md"
FIRM_PROFILE = "Stellenbeschreibung.txt"
TASKS = "Aufgaben.txt"
RULES = "Regeln.md"
OUTPUT = "output/"
QUALIFICATIONS = "Anforderungen.txt"
OLLAMA_URL = "http://ollama:11434"

MODEL_HUGE = "gemma4:e4b"
MODEL_LARGE = "llama3.2:3b"
MODEL_LIGHT = "llama3.2:3b"
MODEL_MIDDLE = "mistral"
GOOGLE_MODEL = "gemini-2.5-flash"
MISTRAL_MODEL = "mistral-large-latest"  # man kann auch andere Modelle wie "mistral-small-latest" wählen
GROQ_MODEL = "groq/openai/gpt-oss-20b" #"meta-llama/llama-4-scout-17b-16e-instruct"
TIMEOUT = 5

#Eigene Unterlagen laden
docs = TextLoader(CV).load()
#Firmenunterlagen laden
firm = TextLoader(FIRM_PROFILE).load()
tasks = TextLoader(TASKS).load()
qual = TextLoader(QUALIFICATIONS).load()

#MarkdownHeaderTextSplitter konfigurieren
headers_to_split_on = [
    ("#", "Kategorie"),
    ("##", "Unterkategorie"),
    ("###", "Abschnitt"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=True  # Header im Chunk-Inhalt nicht behalten
)

# Zuerst nach Headern splitten
text = ""
for doc in docs:
    text += doc.page_content
header_splits = markdown_splitter.split_text(text)

#Rekursiven Splitter für sehr lange Abschnitte
child_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n", ". "," ",""],
    chunk_size=400, 
    chunk_overlap=80)
final_chunks = child_splitter.split_documents(header_splits)

#Embeddings generieren & FAISS-Vektorspeicher erstellen
emb = OllamaEmbeddings(model='mxbai-embed-large', base_url=OLLAMA_URL)
VECTOR_STORE = FAISS.from_documents(final_chunks, emb)

firm_splits = child_splitter.split_documents(firm)

task_splits = child_splitter.split_documents(tasks)

qual_splits = child_splitter.split_documents(qual)


def extract_questions(text: str) -> list[str]:
    """
    Extrahiert alle Fragen aus einem Text.
    
    Args:
        text: Der Eingabetext
        
    Returns:
        Liste der gefundenen Fragen (als Zeilen)
    """
    # Regex: Findet Sätze, die mit einem Fragezeichen enden
    # \w+ : Mindestens ein Wortzeichen für den Anfang
    # [^?]* : Beliebig viele Zeichen, die keine Fragezeichen sind
    # \? : Das finale Fragezeichen
    pattern = r'[A-ZÄÖÜ][^?!.]*\?'
    
    # Alternative: Ausführlicherer Regex mit Unterstützung für Anführungszeichen
    # pattern = r'(?:^|(?<=[.!?]\s))["\']?[A-Z][^?!\n]*\?["\']?'
    
    questions = re.findall(pattern, text, re.MULTILINE)
    
    # Entferne führende/nachfolgende Leerzeichen
    questions = [q.strip() for q in questions]
    
    return questions

class Plan(BaseModel):
    """Fragen, die man dem Bewerber stellen muss"""

    steps: list[str] = Field( 
        description="unterschiedliche Fragen an den Bewerber, müssen sortiert sein"
    )

system_prompt_template = (
    "Du bist ein professioneller Vermittler von qualifizierten Fachkräften im IT-Bereich.\n"
    "Du hilfst dem Bewerber beim Erstellen seines Anschreibens.\n"
    "Stelle dem Bewerber Fragen zum angegebenen Punkt des zu erstellenden Anschreibens, die nur seine Erfahrungen angehen, eine nach der anderen,\n"
    "als eine nichtleere Liste, damit er beim Antworten darauf seine Stärken und Motivation am besten zeigen könnte.\n"
    "Stelle keine gegenstandslosen und überflüssigen Fragen und beantworte sie selbst NICHT. \n"
    "Im Kommentar gibt es eine Beschreibung des Punkts des Anschreibens. \n"
    "Im Kontext1 gibt es Anforderungen der Firma an den Bewerber. \n"
    "Im Kontext2 gibt es seine künftigen Aufgaben. \n"
    "Im Kontext3 gibt es eine kurze Beschreibung der Firma. \n"
)
planner_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template),
     ("user", "Erstelle einen Fragebogen als eine Liste von Fragen zum PUNKT DES ANSCHREIBENS:\n{item}\nKOMMENTAR:\n{comm}\nKONTEXT1:\n{qual}\nKONTEXT2:\n{tasks}\nKONTEXT3:\n{firm}\n")])

llm_small = ChatGroq(model=GROQ_MODEL, temperature=0)
llm_small_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0)   
llm_small_loc = ChatOllama(model=MODEL_LARGE, temperature=0, base_url=OLLAMA_URL)
llm_large = ChatGroq(model=GROQ_MODEL, temperature=0)
llm_large_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0)
llm_large_loc = ChatOllama(model=MODEL_LARGE, temperature=0, base_url=OLLAMA_URL)
llm_huge = ChatGroq(model=GROQ_MODEL, temperature=0.2)
llm_huge_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0.2)   
llm_huge_loc = ChatOllama(model=MODEL_LARGE, temperature=0.2, base_url=OLLAMA_URL)
llm_enorm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0)
llm_enorm_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0)   
llm_enorm_loc = ChatOllama(model=MODEL_HUGE, temperature=0, base_url=OLLAMA_URL)
llm_middle = ChatOllama(model=MODEL_MIDDLE, temperature=0.1, base_url=OLLAMA_URL)
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
llm_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0.2)   
llm_loc = ChatOllama(model=MODEL_LIGHT, temperature=0.2, base_url=OLLAMA_URL)
llm_alt1 = ChatOllama(model=MODEL_MIDDLE, temperature=0.2, base_url=OLLAMA_URL)

planner = planner_prompt | llm_huge | StrOutputParser()   
planner_alt = planner_prompt | llm_huge_alt | StrOutputParser() 
planner_alt1 = planner_prompt | llm_middle | StrOutputParser() 
planner_loc = planner_prompt | llm_huge_loc | StrOutputParser() 

research_tools = load_tools(
  tool_names=["wikipedia"],
  llm=llm_small
)

system_prompt = (
    "Es geht um eine Erstellung eines Punkts eines Anschreibens, wozu eine Frage gestellt wurde. \n"             
    "Benutze die zur Verfügung stehenden Tools **NUR** um nach **Erklärungen zur Art der Beantwortung** "
    "der Frage zu diesem Punkt zu suchen. \n"
)

raw_prompt_template = (
    "Du bist ein Bewerber um eine IT-Stelle und musst eine Frage zu einem Punkt deines Anschreibens beantworten. \n"
    "Dir ist folgendes gegeben:\n"
    "- Frage zu einem Punkt deines Anschreibens\n "
    "- Punkt deines Anschreibens, zu dem die Frage gestellt ist\n "  
    "- Beispiel einer Antwort auf die Frage\n "           
    "- Kommentar zu diesem Punkt, der Anweisungen zum Beantworten der Frage enthält\n "
    "- Vorhergehende Fragen zu diesem Punkt mit Antworten darauf\n "
    "- Kontext mit Dokumenten über deine Arbeitserfahrungen, Ausbildung, Fähigkeiten und Fertigkeiten.\n "
    "Mach keine Annahmen, schreibe NUR auf Basis dieses Kontextes.\n"               
    "Schreib im **Ich-Stil**.\nGib **NUR** die **fertige korrigierte Antwort**, **OHNE Entschuldigungen und / oder Erklärungen**."
    "FRAGE:\n{question}\nPUNKT DES ANSCHREIBENS:\n{item}\nBEISPIEL:\n{example}\nKOMMENTAR:\n{options}\nVORHERGEHENDE FRAGEN MIT ANTWORTEN:\n{previous_steps}\nKONTEXT:\n{documents}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("user", raw_prompt_template),
     ("placeholder", "{messages}")
     ]
)

class ResearchState(AgentState):
    item: str
    question: str
    previous_steps: str
    example: str
    options: str
    documents: str

research_agent = create_react_agent(model=llm_small, tools=research_tools, state_schema=ResearchState, prompt=prompt)
research_agent_alt = create_react_agent(model=llm_small_alt, tools=research_tools, state_schema=ResearchState, prompt=prompt)
research_agent_alt1 = create_react_agent(model=llm_middle, tools=research_tools, state_schema=ResearchState, prompt=prompt)
research_agent_loc = create_react_agent(model=llm_small_loc, tools=research_tools, state_schema=ResearchState, prompt=prompt)

raw_prompt_template_with_critique = (
    "Du bist ein professioneller Vermittler von IT-Fachkräften und du hilfst einem Bewerber im IT-Bereich, "
    "der eine Frage zu einen Punkt seines Anschreibens beantwortet hatte. "
    "Er hat anschließend ein Feedback von einem HR-Mitarbeiter der Firma erhalten, wobei er sich bewirbt. \n"
    "Korrigiere die Antwort des Bewerbers unter Rücksicht auf dieses Feedback und den Kontext. \n"
    "\nFRAGE:\n{question}\nPUNKT DES ANSCHREIBENS:\n{item}\nVORHERGEHENDE FRAGEN MIT ANTWORTEN:\n{previous_steps}\nKONTEXT:\n{documents}\n\n"
    "ANTWORT DES BEWERBERS:\n{answer}\n\nFEEDBACK:\n{feedback}"
    "\nSchreib im **Ich-Stil**. \nGib **NUR** die **fertige korrigierte Antwort**, **OHNE Entschuldigungen und / oder Erklärungen**."
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("user", raw_prompt_template_with_critique),
     ("placeholder", "{messages}")
     ]
)

class ReflectionState(ResearchState):
    answer: str
    feedback: str

research_agent_with_critique = create_react_agent(model=llm_small, tools=research_tools, state_schema=ReflectionState, prompt=prompt)
research_agent_with_critique_alt = create_react_agent(model=llm_small_alt, tools=research_tools, state_schema=ReflectionState, prompt=prompt)
research_agent_with_critique_alt1 = create_react_agent(model=llm_middle, tools=research_tools, state_schema=ReflectionState, prompt=prompt)
research_agent_with_critique_loc = create_react_agent(model=llm_small_loc, tools=research_tools, state_schema=ReflectionState, prompt=prompt)

reflection_prompt = (
    "Du bist ein professioneller HR-Mitarbeiter einer IT-Firma und du schätzst eine Antwort eines Bewerbers ein, "
    "die eine Frage zu einem Punkt seines Anschreibens beantwortet. "
    "\nFRAGE: {question}.\nPUNKT DES ANSCHREIBENS:\n{item}\n."
    "ANTWORT:\n{answer}\n"
    "KONTEXT:\n{documents}\n"
    "KOMMENTAR:\n{options}\n"
    "Denk darüber nach, ob die Antwort gut ist, oder nicht, "
    "und gib ein Feedback mit umsetzbarer Kritik, wenn die Antwort nicht gut ist. "
    "Wenn die Antwort ziemlich gut ist, antworte mit "
    "dem Text der Antwort und gib keine Kritik. Gib deine Kritik nur dann, wenn du denkst, dass die Antwort dem Kontext nicht entspricht, "
    "keine Bestätigung dort findet, dem Kommentar nicht entspricht, oder unlogisch ist. "
    "Mach keine Annahmen. "
)

class Response(BaseModel):
    """Eine endgültige Antwort dem Benutzer."""
    answer: Optional[str] = Field(
        description="Die endgültige Antwort. Sie soll leer sein, wenn es Kritik gibt.",
        default=None,
    )
    critique: Optional[str] = Field(
        description="Kritik zur ursprünglichen Antwort. Wenn du glaubst, dass sie mangelhaft ist, gib ein umsetzbares Feedback",
        default=None,
    )

class ReflectionAgentState(TypedDict):
    item: str
    question: str
    previous_steps: str
    example: str
    options: str
    answer: str
    steps: Annotated[int, add]
    response: Response
    documents: str
    doc_set: set[str]

hypo_llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
hypo_llm_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0.2)   
hypo_llm_alt1 = ChatOllama(model=MODEL_MIDDLE, temperature=0.2, base_url=OLLAMA_URL)
hypo_llm_loc = ChatOllama(model=MODEL_LIGHT, temperature=0.2, base_url=OLLAMA_URL)

def _hypo_doc_embedding(state: ReflectionAgentState):
    # ein Prompt zum Generieren eines hypothetischen Dokuments
    hyde_template = """Auf Basis der Frage: {question}
    zum Punkt: {item} 
    eines Anschreibens
    schreib mit Hilfe von Zusatzinformationen: {options} 
    einen Absatz, der eine Antwort auf diese Frage enthalten könnte.
    Schreib im **Ich-Stil**.
    Gib NUR die fertige Antwort.
    """

    hyde_prompt = PromptTemplate(
        input_variables=["question", "item", "options"],
        template=hyde_template
    )
   
    hyde_chain = hyde_prompt | hypo_llm | StrOutputParser()
    hyde_chain_alt = hyde_prompt | hypo_llm_alt | StrOutputParser()
    hyde_chain_alt1 = hyde_prompt | hypo_llm_alt1 | StrOutputParser()    
    hyde_chain_loc = hyde_prompt | hypo_llm_loc | StrOutputParser()
    print(state)
    # hypotheticsches Dokument generieren
    query = {"question": state["question"],
             "item": state["item"],
             "options": state["options"]}
    hypothetical_doc = []
    try:
        hypothetical_doc = hyde_chain.invoke(query)
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            hypothetical_doc = hyde_chain_loc.invoke(query)
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                hypothetical_doc = hyde_chain_alt.invoke(query)
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                hypothetical_doc = hyde_chain_alt1.invoke(query)               
       
    print("############### hypothetical doc ###############")
    print(hypothetical_doc)
    print ("\nooooooooooooooooooooooooooo\n")
    # Benutze das hypothetisce Dokument zum Auslesen
    embeddings = OllamaEmbeddings(model='mxbai-embed-large', base_url=OLLAMA_URL)
    embedded_query = embeddings.embed_query(hypothetical_doc)
    results = VECTOR_STORE.similarity_search_by_vector(embedded_query, k=7)

    docs = [ doc.metadata.get('Unterkategorie', '-') + ": " + doc.page_content for doc in results ]
    doc_set = {"empty"}
    doc_set.clear() 
    for dc in docs:
        doc_set.add(dc)
       
    print("############### hypo-docs ###############")
    for dc in doc_set:
        print(dc)
        print ("\nooooooooooooooooooooooooooo\n")   
    return {"documents": "\n".join(doc_set), "doc_set": doc_set, "example": hypothetical_doc}  

def _hypo_doc_embedding_ext(state: ReflectionAgentState):
    # ein Prompt zum Generieren eines hypothetischen Dokuments mit Kritik
    hyde_template = """Auf Basis der Frage: {question}
    zum Punkt: {item} 
    eines Anschreibens
    schreib mit Hilfe von Zusatzinformationen wie
    ehemalige Antwort: {answer} und Kritik darauf: {critique} 
    einen Absatz, der eine korrigierte Antwort auf diese Frage enthalten könnte.
    Schreib im **Ich-Stil**.
    Gib NUR die fertige Antwort.
    """

    hyde_prompt = PromptTemplate(
        input_variables=["question", "item", "options"],
        template=hyde_template
    )

    hyde_chain = hyde_prompt | hypo_llm | StrOutputParser()
    hyde_chain_alt = hyde_prompt | hypo_llm_alt | StrOutputParser()
    hyde_chain_alt1 = hyde_prompt | hypo_llm_alt1 | StrOutputParser()
    hyde_chain_loc = hyde_prompt | hypo_llm_loc | StrOutputParser()
    print(state)
    # hypotheticsches Dokument generieren
    query = {"question": state["question"],
             "item": state["item"],
             "answer": state["answer"],
             "critique": state["response"].critique if state.get("response") and state["response"].critique else ""}
    hypothetical_doc = []
    try:
        hypothetical_doc = hyde_chain.invoke(query)
    except:
        try:
            print("******* no cloud model, try a local one *******\n")
            hypothetical_doc = hyde_chain_loc.invoke(query)
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                hypothetical_doc = hyde_chain_alt.invoke(query)
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                hypothetical_doc = hyde_chain_alt1.invoke(query) 
    print("############### hypothetical doc extended ###############")
    print(hypothetical_doc)
    print ("\nooooooooooooooooooooooooooo\n")
    # Benutze das hypothetisce Dokument zum Auslesen
    embeddings = OllamaEmbeddings(model='mxbai-embed-large', base_url=OLLAMA_URL)
    embedded_query = embeddings.embed_query(hypothetical_doc)
    results = VECTOR_STORE.similarity_search_by_vector(embedded_query, k=7)

    docs = [ doc.metadata.get('Unterkategorie', '-') + ": " + doc.page_content for doc in results ]

    doc_set = state["doc_set"]

    for dc in docs:
        doc_set.add(dc)
       
    print("############### hypo-docs-ext ###############")
    for dc in doc_set:
        print(dc)
        print ("\nooooooooooooooooooooooooooo\n")   
    return {"documents": "\n".join(doc_set), "doc_set": doc_set}    

def _should_end(state: ReflectionAgentState, config: RunnableConfig) -> Literal["ext_fill_doc", END]:
    max_reasoning_steps = config["configurable"].get("max_reasoning_steps", 1)
    if state.get("response") and state["response"].answer:
        return END
    if state.get("steps", 1) > max_reasoning_steps:
        return END
    return "ext_fill_doc"

def _should_end1(state: ReflectionAgentState, config: RunnableConfig) -> Literal["research", END]:
    max_reasoning_steps = config["configurable"].get("max_reasoning_steps", 1)
    if state.get("response") and state["response"].answer:
        return END
    if state.get("steps", 1) > max_reasoning_steps:
        return END
    return "research"

def _should_end2(state: ReflectionAgentState, config: RunnableConfig) -> Literal["reflect", END]:
    max_reasoning_steps = config["configurable"].get("max_reasoning_steps", 1)
    if state.get("response") and state["response"].answer:
        return END
    if state.get("steps", 0) >= max_reasoning_steps:
        return END
    return "reflect"


reflection_chain = PromptTemplate.from_template(reflection_prompt) | llm | StrOutputParser()  
reflection_chain_alt = PromptTemplate.from_template(reflection_prompt) | llm_alt | StrOutputParser()
reflection_chain_alt1 = PromptTemplate.from_template(reflection_prompt) | llm_alt1 | StrOutputParser()
reflection_chain_loc = PromptTemplate.from_template(reflection_prompt) | llm_loc | StrOutputParser()

def _reflection_step(state: ReflectionAgentState):
    print(f"\n==== Antwort: {state["answer"]} ====\n")
    result = []
    try:
        result = reflection_chain.invoke(state)
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            result = reflection_chain_loc.invoke(state)
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                result = reflection_chain_alt.invoke(state)
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                result = reflection_chain_alt1.invoke(state)        
          
    res = []
    
    if state["answer"] in result:
       res = Response(answer=result)
       print(f"\n++++ Korrekte Antwort: {result} ++++\n")
    else:
       res = Response(critique=result)
       print(f"\n---- Kritik: {result} ----\n")
    if not result:
       res = Response(answer=result)
       print(f"\n**** kein Ergebnis, {result} ****\n")
    return {"response": res, "steps": 1}


def _research_start(state: ReflectionAgentState):
    answer = []
    try:
        answer = research_agent.invoke(state)
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            answer = research_agent_loc.invoke(state)
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                answer = research_agent_alt.invoke(state)
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                answer = research_agent_alt1.invoke(state)
    return {"answer": answer["messages"][-1].content}


def _research(state: ReflectionAgentState):
    agent_state = {
      "item": state["item"],
      "answer": state["answer"],
      "question": state["question"],
      "previous_steps": state["previous_steps"],
      "documents": state["documents"],
      "options": state["options"],
      "feedback": state["response"].critique
    }
    answer = []
    try:
        answer = research_agent_with_critique.invoke(agent_state)
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            answer = research_agent_with_critique_loc.invoke(agent_state)
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                answer = research_agent_with_critique_alt.invoke(agent_state)
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                answer = research_agent_with_critique_alt1.invoke(agent_state)
    print(f"\n#### revidierte Antwort: {answer["messages"][-1].content} ####\n")
    return {"answer": answer["messages"][-1].content}


builder = StateGraph(ReflectionAgentState)
builder.add_node("fill_documents", _hypo_doc_embedding)
builder.add_node("research_start", _research_start)
builder.add_node("research", _research)
builder.add_node("reflect", _reflection_step)
builder.add_node("ext_fill_doc", _hypo_doc_embedding_ext)

builder.add_edge(START, "fill_documents")
builder.add_edge("fill_documents", "research_start")
builder.add_edge("research_start", "reflect")
builder.add_conditional_edges("reflect", _should_end2)
builder.add_edge("ext_fill_doc", "research")
builder.add_conditional_edges("reflect", _should_end)
hypo_answer = builder.compile()

class AnswerArgs(BaseModel):
    item: str = Field(description="Punkt des Anschreibens, wozu die angegebene Frage beantwortet werden muss")
    question: str = Field(description="Frage zu beantworten")
    previous_steps: str = Field(description="vorhergehende Fragen zum angegebenen Punkt des Anschreibens mit Antworten darauf")
    options: str = Field(description="Kommentar zum Punkt des Anschreibens, der eine Erklärung zum Beantworten enthält")

class TreeNode:

    def __init__(
        self,
        node_id: int,
        step: str,
        step_output: Optional[str] = None,
        parent: Optional["TreeNode"] = None,
    ):
        self.node_id = node_id
        self.step = step
        self.step_output = step_output
        self.parent = parent
        self.children = []
        self.final_response = None

    def __repr__(self):
        parent_id = self.parent.node_id if self.parent else "None"
        return f"Node_id: {self.node_id}, parent: {parent_id}, {len(self.children)} children."

    def get_full_plan(self) -> str:
        """Returniert formatierte Frageliste mit Fragenummern und übergebenen Antworten."""
        steps = []
        node = self
        while node.parent:
            steps.append((node.step, node.step_output))
            node = node.parent

        full_plan = []
        for i, (step, result) in enumerate(steps[::-1]):
            if result:
                full_plan.append(f"# {i+1}. Frage: {step}\nAntwort: {result}\n")
        return "\n".join(full_plan)


class PlanState(TypedDict):
    item: str
    comm: str
    firm: list[str]
    tasks: list[str]
    qual: list[str]
    root: TreeNode
    queue: deque[TreeNode]
    current_node: TreeNode
    next_node: TreeNode
    is_current_node_final: bool
    paths_explored: Annotated[int, operator.add]
    visited_ids: set[int]
    max_id: int
    candidates: Annotated[list[str], operator.add]
    best_candidate: str

class ReplanStep(BaseModel):
    """Umgeänderte nächste Frage im Fragebogen."""

    steps: list[str] = Field(
        description="andere Optionen nächster vorgeschlagener Frage"
    )

llm_replanner = llm_huge  | StrOutputParser()  
llm_replanner_alt = llm_huge_alt  | StrOutputParser()
llm_replanner_alt1 = llm_middle  | StrOutputParser()
llm_replanner_loc = llm_huge_loc  | StrOutputParser()

replanner_prompt_template = (
    "Überlege dir die nächste Frage im Fragebogen. \n"
    "Returniere diese Frage als eine Liste mit einer Zeile. \n"
    "Wenn du denkst, dass keine Frage mehr nötig ist, returniere einfach eine leere Liste []. \n"
    "PUNKT DES ANSCHREIBENS: {item}\n VORHERGEHENDE FRAGEN MIT ANTWORTEN: {current_plan} \n"
    "KOMMENTAR:\n{comm}\nKONTEXT1:\n{qual}\nKONTEXT2:\n{tasks}\nKONTEXT3:\n{firm}\n"
)

replanner_prompt = ChatPromptTemplate.from_messages(
    [("system", """Du bist ein professioneller Vermittler von qualifizierten Fachkräften im IT-Bereich.
Du hilfst dem Bewerber beim Erstellen seines Anschreibens.
Stelle dem Bewerber Fragen zum angegebenen Punkt des zu erstellenden Anschreibens, eine nach der anderen,
als eine Liste **list[str]**, damit er beim Antworten darauf seine Stärken und Motivation am besten zeigen könnte.
Stelle keine gegenstandslosen und überflüssigen Fragen und beantworte sie selbst NICHT. 
Im Kommentar gibt es eine Beschreibung des Punkts des Anschreibens. 
Im Kontext1 gibt es Anforderungen der Firma an den Bewerber. 
Im Kontext2 gibt es seine künftigen Aufgaben. 
Im Kontext3 gibt es eine kurze Beschreibung der Firma. 
      """),
     ("user", replanner_prompt_template)
    ]
)

replanner = replanner_prompt | llm_replanner
replanner_alt = replanner_prompt | llm_replanner_alt
replanner_alt1 = replanner_prompt | llm_replanner_alt1
replanner_loc = replanner_prompt | llm_replanner_loc

prompt_voting = PromptTemplate.from_template(
    "Du bist ein professioneller HR-Mitarbeiter einer IT-Firma und schätzst Antworte von Bewerbern ein, "
    "die eine Frage zu einem Punkt ihres Anschreibens beantwortet haben. \n" 
    "Wähle die beste Antwort für den Punkt des Anschreibens. "
    "Im Kommentar gibt es eine Beschreibung des Punktes. \n"
    "Im Kontext1 gibt es Anforderungen der Firma an den Bewerber. \n"
    "Im Kontext2 gibt es seine künftigen Aufgaben. \n"
    "Im Kontext3 gibt es eine kurze Beschreibung der Firma. \n"
    "PUNKT DES ANSCHREIBENS:{item}\n\nANTWORTE:\n{candidates}\n"
     "\nKOMMENTAR:\n{comm}\nKONTEXT1:\n{qual}\nKONTEXT2:\n{tasks}\nKONTEXT3:\n{firm}\n"
)

def _vote_for_the_best_option(state: PlanState):
    candidates = state.get("candidates", [])
    if not candidates:
        return {"best_response": None}
    all_candidates = []
    for i, candidate in enumerate(candidates):
        all_candidates.append(f"OPTION {i+1}: {candidate}")
  
    response_schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "enum": [str(i+1) for i in range(len(all_candidates))]
            }
        },
        "required": ["value"]
    }
    llm_enum = ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0.1, format=response_schema)   
    llm_enum_alt = ChatMistralAI(model=MISTRAL_MODEL, temperature=0.1, format=response_schema)
    llm_enum_loc = ChatOllama(model=MODEL_HUGE, temperature=0.1, base_url=OLLAMA_URL, format=response_schema) 
    llm_enum_alt1 = ChatOllama(model=MODEL_MIDDLE, temperature=0.1, base_url=OLLAMA_URL, format=response_schema)
    result = []
    try:
        result = (prompt_voting | llm_enum).invoke(
            {"candidates": "\n".join(all_candidates), "item": state["item"],
            "comm": state["comm"], "qual": "\n".join(state["qual"]),
            "tasks": "\n".join(state["tasks"]), "firm": "\n".join(state["firm"])}
        )
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            result = (prompt_voting | llm_enum_loc).invoke(
                {"candidates": "\n".join(all_candidates), "item": state["item"],
                "comm": state["comm"], "qual": "\n".join(state["qual"]),
                "tasks": "\n".join(state["tasks"]), "firm": "\n".join(state["firm"])}
            ) 
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                result = (prompt_voting | llm_enum_alt).invoke(
                    {"candidates": "\n".join(all_candidates), "item": state["item"],
                    "comm": state["comm"], "qual": "\n".join(state["qual"]),
                    "tasks": "\n".join(state["tasks"]), "firm": "\n".join(state["firm"])}
                )
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                result = (prompt_voting | llm_enum_alt1).invoke(
                    {"candidates": "\n".join(all_candidates), "item": state["item"],
                    "comm": state["comm"], "qual": "\n".join(state["qual"]),
                    "tasks": "\n".join(state["tasks"]), "firm": "\n".join(state["firm"])}
                )                
                             
    print("\n------------\n".join(all_candidates))
    print("\n\n\n")
    print(result.content)
    res = {"value": "1"}
    var = result.content
    # Typ prüfen
    if isinstance(var, str):
        try:
            # String sicher auswerten
            ergebnis = json.loads(var)
            # Prüfen, ob das Ergebnis wirklich ein Dictionary ist
            if isinstance(ergebnis, dict):
                res = ergebnis
            else:
                print("Der String repräsentiert kein Dictionary")
        except (ValueError, SyntaxError):
            print("Ungültiger Syntax für ein Dictionary")
            prompt = """
            Finde aus dem Text heraus, welche OPTION ausgewählt wurde.
            Gib nur diese OPTION als eine ganze Zahl wie 1, 2 oder 3 aus.

            TEXT: 
            {original_text}

            OPTION: 
            """

            clean_input = prompt.format(original_text=var)
            clean_output = []
            try:
                clean_output = llm_enorm.invoke(clean_input)
            except:
                time.sleep(TIMEOUT)
                try:
                    print("******* no cloud model, try a local one *******\n")
                    clean_output = llm_enorm_loc.invoke(clean_input)
                except:
                    time.sleep(TIMEOUT)
                    try:
                        print("??????? no local model, try an alternative one ???????\n")
                        clean_output = llm_enorm_alt.invoke(clean_input)
                    except:
                        time.sleep(TIMEOUT)
                        print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                        clean_output = llm_middle.invoke(clean_input)
                
            int_res = clean_output.content.strip()
            try:
                if int(int_res) > 0:
                    res = {"value": str(int_res)}
                else:
                    print("OPTION - keine positive Ganzzahl: " + str(int_res)) 
            except:
                print("OPTION - keine Ganzzahl: " + str(int_res)) 

    elif isinstance(var, dict):
        print("Variable ist kein String sondern Dict")
        res = var
    else:
        print("Variable ist " + str(type(var)))    


    print("\n\n\n++++++++++") 
    print(candidates[int(res["value"])-1])
    print("\n\n\n=================\n") 
    return {"best_candidate": candidates[int(res["value"])-1]}

final_prompt = PromptTemplate.from_template(
    "Du bist ein Bewerber um eine IT-Stelle und hast den Fragebogen ausgefüllt."
    "Mach eine Zusammenfassung des Fragebogens, die dem Punkt des Anschreibens klaren Inhalt geben würde.\n"
    "Mach keine Annahmen, schreib NUR auf Basis des Fragebogens.\n"
    "Schreib im **Ich-Stil**. Gib NUR die Zusammenfassung.\n"
    "Im Kommentar gibt es eine Beschreibung des Punkts des Anschreibens. \n"
    "PUNKT DES ANSCHREIBENS:\n{item}\n\nFRAGEBOGEN:\n{plan}\n\nKOMMENTAR:\n{comm}\n"
    "ZUSAMMENFASSUNG ZUM PUNKT DES ANSCHREIBENS:\n"
)

responder = final_prompt | llm_large | StrOutputParser() 
responder_alt = final_prompt | llm_large_alt | StrOutputParser()
responder_alt1 = final_prompt | llm_middle | StrOutputParser()
responder_loc = final_prompt | llm_large_loc | StrOutputParser()  

async def _build_initial_plan(state: PlanState) -> PlanState:
    plan_raw = []
    try:
        plan_raw = await planner.ainvoke({"item": state["item"],
                                "comm": state["comm"],
                                "qual": "\n".join(state["qual"]),
                                "tasks": "\n".join(state["tasks"]),
                                "firm": "\n".join(state["firm"])})
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            plan_raw = await planner_loc.ainvoke({"item": state["item"],
                                "comm": state["comm"],
                                "qual": "\n".join(state["qual"]),
                                "tasks": "\n".join(state["tasks"]),
                                "firm": "\n".join(state["firm"])})  
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                plan_raw = await planner_alt.ainvoke({"item": state["item"],
                                "comm": state["comm"],
                                "qual": "\n".join(state["qual"]),
                                "tasks": "\n".join(state["tasks"]),
                                "firm": "\n".join(state["firm"])}) 
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                plan_raw = await planner_alt1.ainvoke({"item": state["item"],
                                "comm": state["comm"],
                                "qual": "\n".join(state["qual"]),
                                "tasks": "\n".join(state["tasks"]),
                                "firm": "\n".join(state["firm"])})                               
    plan = Plan(steps=extract_questions(plan_raw))
    queue = deque()
    root = TreeNode(step=plan.steps[0], node_id=1)
    queue.append(root)
    current_root = root
    for i, step in enumerate(plan.steps[1:]):
        child = TreeNode(node_id=i+2, step=step, parent=current_root)
        current_root.children.append(child)
        queue.append(child)
        current_root = child
    return {"root": root, "queue": queue, "max_id": i+2}

async def _run_node(state: PlanState, config: RunnableConfig):
    node = state.get("next_node")
    visited_ids = state.get("visited_ids", set())
    queue = state["queue"]
    if node is None:
        while queue and not node:
            node = state["queue"].popleft()
            if node.node_id in visited_ids:
                node = None
        if not node:
            return Command(goto="vote", update={})

    step = await hypo_answer.ainvoke({
        "previous_steps": node.get_full_plan(),
        "question": node.step,
        "item": state["item"],
        "options": state["comm"]
    })
    node.step_output = step["answer"]

    visited_ids.add(node.node_id)
    return {"current_node": node, "queue": queue, "visited_ids": visited_ids, "next_node": None}

async def _plan_next(state: PlanState, config: RunnableConfig) -> PlanState:
    max_candidates = config["configurable"].get("max_candidates", 1)
    node = state["current_node"]
    next_step_raw = []
    try:
        next_step_raw = await replanner.ainvoke({"item": state["item"], "current_plan": node.get_full_plan(), 
                "comm": state["comm"],
                "qual": "\n".join(state["qual"]),
                "tasks": "\n".join(state["tasks"]),
                "firm": "\n".join(state["firm"])})
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            next_step_raw = await replanner_loc.ainvoke({"item": state["item"], "current_plan": node.get_full_plan(), 
                "comm": state["comm"],
                "qual": "\n".join(state["qual"]),
                "tasks": "\n".join(state["tasks"]),
                "firm": "\n".join(state["firm"])})  
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                next_step_raw = await replanner_alt.ainvoke({"item": state["item"], "current_plan": node.get_full_plan(), 
                    "comm": state["comm"],
                    "qual": "\n".join(state["qual"]),
                    "tasks": "\n".join(state["tasks"]),
                    "firm": "\n".join(state["firm"])}) 
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                next_step_raw = await replanner_alt1.ainvoke({"item": state["item"], "current_plan": node.get_full_plan(), 
                    "comm": state["comm"],
                    "qual": "\n".join(state["qual"]),
                    "tasks": "\n".join(state["tasks"]),
                    "firm": "\n".join(state["firm"])})                 

    next_step = ReplanStep(steps=extract_questions(next_step_raw))
    if not next_step.steps:
        return {"is_current_node_final": True}
    max_id = state["max_id"]
    for step in next_step.steps[:max_candidates]:
        child = TreeNode(node_id=max_id+1, step=step, parent=node)
        max_id += 1
        node.children.append(child)
        state["queue"].append(child)
    return {"is_current_node_final": False, "next_node": child, "max_id": max_id}

async def _get_final_response(state: PlanState) -> PlanState:
    node = state["current_node"]
    final_response = []
    try:
        final_response = await responder.ainvoke({"item": state["item"], "plan": node.get_full_plan(), "comm": state["comm"]})
    except:
        time.sleep(TIMEOUT)
        try:
            print("******* no cloud model, try a local one *******\n")
            final_response = await responder_loc.ainvoke({"item": state["item"], "plan": node.get_full_plan(), "comm": state["comm"]})
        except:
            time.sleep(TIMEOUT)
            try:
                print("??????? no local model, try an alternative one ???????\n")
                final_response = await responder_alt.ainvoke({"item": state["item"], "plan": node.get_full_plan(), "comm": state["comm"]})
            except:
                time.sleep(TIMEOUT)
                print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
                final_response = await responder_alt1.ainvoke({"item": state["item"], "plan": node.get_full_plan(), "comm": state["comm"]})
    node.final_response = final_response
    print("xxxxxxxxxxxx final_response xxxxxxxxxxx")
    print(final_response)
    print ("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")  
    return {"paths_explored": 1, "candidates": [final_response]}

def _should_create_final_response(state: PlanState) -> Literal["run", "generate_response"]:
    return "generate_response" if state["is_current_node_final"] else "run"

def _should_continue(state: PlanState, config: RunnableConfig) -> Literal["run", "vote"]:
    max_paths = config["configurable"].get("max_paths", 30)
    if state.get("paths_explored", 1) >= max_paths:
        return "vote"
    if state["queue"] or state.get("next_node"):
        return "run"
    return "vote"

builder = StateGraph(PlanState)
builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_node)
builder.add_node("plan_next", _plan_next)
builder.add_node("generate_response", _get_final_response)
builder.add_node("vote", _vote_for_the_best_option)

builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_edge("run", "plan_next")
builder.add_conditional_edges("plan_next", _should_create_final_response)
builder.add_conditional_edges("generate_response", _should_continue)
builder.add_edge("vote", END)

graph = builder.compile()

async def graph_invoker(graph,item,comm,firm,tasks,qual):
    result = await graph.ainvoke({"item": item, "comm": comm, 
        "firm": firm, "tasks": tasks, "qual": qual}, 
        config={"recursion_limit": 1000, "configurable": {"max_paths": 3}})
    return result

async def main(hd_splits,graph,firm_splits,task_splits,qual_splits) -> str:
    cover_text = ""
    comm_text = ""
    for rule in hd_splits:
        item = rule.metadata.get('Punkt')
        file_name = OUTPUT + item + '.txt'
        try:
            # Versuche die Datei aufzumachen
            with open(file_name, 'r', encoding="utf-8") as file:
                comm = rule.page_content
                print(f"\nPunkt {item} existiert schon.\n")
                cover_text += "**"+item+"**" + "\n\n" + file.read() + "\n\n"
                comm_text += "**"+item+"**" + "\n\n" + comm
        except FileNotFoundError:
            comm = rule.page_content
            print(f"\n=======\n{item}\n=======\n{comm}\n")
            result = await graph_invoker(graph,item,comm,
                    [doc.page_content for doc in firm_splits],
                    [doc.page_content for doc in task_splits],
                    [doc.page_content for doc in qual_splits])
            print(f"\n=======\n{result['best_candidate']}\n=======\n der beste von {len(result['candidates'])} Kandidaten\n")
            with open(file_name, 'w', encoding="utf-8") as file:
                print(result['best_candidate'], file=file)
                cover_text += "**"+item+"**" + "\n\n" + result['best_candidate'] + "\n\n"
                comm_text += "**"+item+"**" + "\n\n" + comm
    return (cover_text,comm_text)

rules = TextLoader(RULES).load()

#MarkdownHeaderTextSplitter konfigurieren
hd_to_split_on = [
    ("#", "Punkt"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=hd_to_split_on, 
    strip_headers=True  # Header im Chunk-Inhalt nicht behalten
)

# Nach Headern splitten
text = ""
for rule in rules:
    text += rule.page_content
hd_splits = md_splitter.split_text(text)

result_raw, comm_raw = asyncio.run(main(hd_splits,graph,firm_splits,task_splits,qual_splits))
cleaner_prompt = """
Erstelle ein komplettes Anschreiben aus dem angegebenen Text.
Die Regeln dafür sind im angegebenen Kommentar

TEXT: 
{original_text}

KOMMENTAR: 
{original_comm}
"""

clean_input = cleaner_prompt.format(original_text=result_raw,original_comm=comm_raw)
clean_output = []
try:
    clean_output = llm_enorm.invoke(clean_input)
except:
    time.sleep(TIMEOUT)
    try:
        print("******* no cloud model, try a local one *******\n")
        clean_output = llm_enorm_loc.invoke(clean_input)
    except:
        time.sleep(TIMEOUT)
        try:
            print("??????? no local model, try an alternative one ???????\n")
            clean_output = llm_enorm_alt.invoke(clean_input)
        except:
            time.sleep(TIMEOUT)
            print("!!!!!!! no other models, try the alternative two !!!!!!!\n")
            clean_output = llm_middle.invoke(clean_input)
    
result = clean_output.content.strip()
print(f"Das Anschreiben besteht aus folgenden Teilen:\n\n=======\n{result}\n=======\n")
file_name = OUTPUT + 'cover_letter_body.txt'
with open(file_name, 'w', encoding="utf-8") as file:
    print(result, file=file)

