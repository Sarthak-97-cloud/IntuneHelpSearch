import sys
sys.path.append(".")
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
#import sys
#sys.path.append("My_Project")
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Doc
from utils.utils import get_logger, clean_text, truncate_text

logger = get_logger(__name__)

class DocParser:

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in configuration.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        self.output_parser = PydanticOutputParser(pydantic_object=Doc)

        self.prompt_template = PromptTemplate(
            template = """You are an expert technical troubleshooting document parser. Extract the following fields 
            from the text provided.
            
            Troubleshooting Text:
            {troubleshooting_text}

            {format_instructions}

            Return a valid JSON object that matches the document schema exactly.""",
            input_variables = ["troubleshooting_text"],
            partial_variables = {"format_instructions": self.output_parser.get_format_instructions()}
        )

    def parse_doc(self, doc_text: str, doc_id: str) -> Dict[str, Any]:
        cleaned_text = clean_text(doc_text)
        truncated_text = truncate_text(cleaned_text)

        chain = self.prompt_template | self.llm | self.output_parser

        parsed_doc = chain.invoke({"troubleshooting_text": truncated_text})

        result = parsed_doc.model_dump()

        result.update(
            {
                "doc_id": doc_id,
                "text": truncated_text
            }
        )
        return result

if __name__ == "__main__":
    parser = DocParser()
    #parsed = parser.parse_doc(sample_doc, doc_id="12345")