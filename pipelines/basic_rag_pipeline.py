from typing import Any

class BasicRAGPipeline:

    def __init__(
        self,
        retriever,
        llm,
        prompt_manager,
        evaluator=None
    ):
        
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.evaluator = evaluator

    def retrieve(
        self,
        question: str
    ) -> list[str]:
        
        return self.retriever.retrieve(question)
    
    def generate(
        self,
        question: str
    ) -> dict[str, Any]:
        
        contexts = self.retrieve(question)

        context = "\n\n".join(contexts)

        prompt = self.prompt_manager.render(
            "rag/basic.jinja2",
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "contexts": contexts,
            "answer": answer
        }
    

    def evaluate(
        self,
        prediction,
        reference
    ):
        
        if (self.evaluator is None):
            raise ValueError("Evaluator not provided")
        
        return self.evaluator.compute(
            predictions=[prediction],
            references=[reference]
        )