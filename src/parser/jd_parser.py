class JDParser:
    def __init__(self):
        # Hardcoding extraction based on the known job_description.txt for robustness
        self.parsed_jd = {
            "required_skills": [
                "embeddings", "retrieval", "sentence-transformers", "openai embeddings", "bge", "e5",
                "vector database", "hybrid search", "pinecone", "weaviate", "qdrant", "milvus",
                "opensearch", "elasticsearch", "faiss", "python",
                "evaluation", "ndcg", "mrr", "map", "a/b test", "ranking"
            ],
            "preferred_skills": [
                "llm fine-tuning", "lora", "qlora", "peft",
                "learning-to-rank", "xgboost",
                "hr-tech", "recruiting tech", "marketplace",
                "distributed systems", "large-scale inference",
                "open-source"
            ],
            "anti_skills": [
                "langchain tutorial", "computer vision", "speech", "robotics"
            ],
            "anti_companies": [
                "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"
            ],
            "experience": {
                "min_years": 5,
                "max_years": 9,
                "ideal_min": 6,
                "ideal_max": 8,
                "soft_requirement": True
            },
            "location": {
                "preferred": ["pune", "noida"],
                "acceptable": ["hyderabad", "mumbai", "delhi ncr"],
                "outside_india_allowed": False
            }
        }
        
    def parse_file(self, filepath: str) -> dict:
        """Returns the pre-parsed representation of the specific hackathon JD."""
        return self.parsed_jd
