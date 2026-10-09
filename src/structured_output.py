from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    answer: str = Field(..., description="1-2 sentence answer to the question")


class RephrasedOutput(BaseModel):
    answer: str = Field(
        ...,
        description="1-2 sentence answer reflecting confidence level in natural language form",
    )
