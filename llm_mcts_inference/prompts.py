from pydantic import BaseModel


class RatingResponse(BaseModel):
    justification: str
    rating: int


class ImprovedResponse(BaseModel):
    ImprovedText: str


critique_prompt = """
You are an expert assistant analyzing the user's original prompt and the provided initial answer.
Your goal is to give clear, constructive, and concise feedback that will guide improvement.

Original Prompt:
{original_prompt}

Initial Answer:
{initial_answer}

Instructions:
- Provide a high-quality critique that focuses on how this answer could be improved.
- Be concise and to the point.
- Highlight key areas that need correction, clarification, or further detail.
- Do not rewrite or provide the full answer; focus only on providing feedback.
"""

refine_prompt = """
You are an expert assistant refining the previous answer based on new feedback.

Original Prompt:
{original_prompt}

Previous Answer:
{previous_answer}

Feedback (Critique):
{feedback}

Instructions:
- Incorporate the provided feedback to enhance correctness, clarity, and completeness.
- Maintain relevance to the original prompt.
- Produce a revised answer that is improved in quality.
"""

rating_prompt = """
You are an expert assistant evaluating the quality of the improved answer.

Original Prompt:
{original_prompt}

Improved Answer:
{improved_answer}

Instructions:
- Assign a rating from 0 to 100, where 0 is completely inadequate and 100 is a perfect response.
- Provide a concise justification (1–3 sentences) explaining your rating.
- Clearly output your final numeric rating on its own line at the end.
"""
