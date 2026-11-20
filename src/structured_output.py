from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    answer: str = Field(..., description="1-2 sentence answer to the question")


class RephrasedOutput(BaseModel):
    answer: str = Field(
        ...,
        description="1-2 sentence answer reflecting confidence level in natural language form",
    )


class PerceivedConfidenceOutput(BaseModel):
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Perceived confidence and reliablility of the wording, from 0 (very hedged) to 100 (fully assured)",
    )
