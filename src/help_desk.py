import sys
import load_db
import collections
from langchain.llms import OpenAI, Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings


class HelpDesk:
    """Create the necessary objects to create a QARetrieval chain"""

    def __init__(self, new_db=True):

        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()

    def get_template(self):
        template = """
        Given this text extracts:
        -----
        {context}
        
        -----
        Please provide the most reliable response possible, and if you're not sure about your answer, say 'Sorry, I don't know'.
        Question: {question}
        Helpful Answer:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )
        return prompt

    def get_embeddings(self):
            return self.get_embeddingsOllama()
        
    def get_llm(self):
            return self.get_llmOllama()
        
    def get_embeddingsOpenAI(self) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return embeddings
    
    def get_embeddingsOllama(self) -> OllamaEmbeddings:
        embeddings = OllamaEmbeddings(model="mistral")
        return embeddings
    
    def get_llmOllama(self):
        llm = Ollama(model="mistral")
        
    def get_llmOpenAi(self):
        llm = OpenAI()
        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        return qa

    def retrieval_qa_inference(self, question, verbose=True):
        query = {"query": question}
        answer = self.retrieval_qa_chain(query)
        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(sources)

        return answer["result"], sources

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["source"]})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[
                0
            ][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"

        else:
            return "Désolé je n'ai trouvé aucune ressource pour répondre à ta question"
