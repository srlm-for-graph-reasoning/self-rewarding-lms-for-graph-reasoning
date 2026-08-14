from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field

PROMPTS_DIR = Path(__file__).parent / "prompts"

CYPHER_GENERATION_CONTENT_PROMPT_FILE = PROMPTS_DIR / "cypher_generation_content_prompt.txt"
CYPHER_GENERATION_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "cypher_generation_system_prompt.txt"
CYPHER_JUDGE_CONTENT_PROMPT_FILE = PROMPTS_DIR / "cypher_judge_content_prompt.txt"
CYPHER_JUDGE_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "cypher_judge_system_prompt.txt"

CYPHER_GENERATION_CONTENT_PROMPT = CYPHER_GENERATION_CONTENT_PROMPT_FILE.read_text()
CYPHER_GENERATION_SYSTEM_PROMPT = CYPHER_GENERATION_SYSTEM_PROMPT_FILE.read_text()
CYPHER_JUDGE_CONTENT_PROMPT = CYPHER_JUDGE_CONTENT_PROMPT_FILE.read_text()
CYPHER_JUDGE_SYSTEM_PROMPT = CYPHER_JUDGE_SYSTEM_PROMPT_FILE.read_text()

class CypherResponse(BaseModel):
    reasoning: str = Field(description="The model's hidden chain of thought extracted from the analysis channel.")
    cypher_query: Optional[str] = Field(default=None, description="The final executable Cypher query.")

class JudgeResponse(BaseModel):
    rationale: str
    score: int