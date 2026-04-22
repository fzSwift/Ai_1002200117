from __future__ import annotations

ADVERSARIAL_QUERIES = [
    "Who won the election there?",
    "What was the 2026 education budget allocation?",
    "What did the minister say about agriculture?",
]

FACTUAL_QUERIES = [
    "What is the title of the 2025 Ghana budget statement?",
    "How many months of gross external reserves did Ghana achieve as of December 2024?",
    "What was the total central government payables as at end 2024?",
]

NUMERIC_QUERIES = [
    "What was the inflation rate for Ghana in 2024 according to the convergence table?",
    "What is the total provisional domestic debt service from 2025 to 2028?",
]

# Deeper, domain-specific queries (election CSV + 2025 budget PDF coverage varies by year).
DEPTH_QUERIES = [
    "Total votes in the 1992 Ghana presidential election",
    "Which region had the highest number of votes for Nana Akufo Addo in 2020?",
    "What is Ghana's inflation target for 2025 according to the budget?",
    "What is the fiscal deficit target for 2025 in the budget statement?",
]

ALL_EVAL_QUERIES = ADVERSARIAL_QUERIES + FACTUAL_QUERIES + NUMERIC_QUERIES + DEPTH_QUERIES
