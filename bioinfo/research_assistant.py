import logging
from typing import List, Dict, Any
from core.base_rag import BaseRAG

class ResearchAssistant:
    def __init__(self, rag: BaseRAG):
        self.rag = rag
        self.logger = logging.getLogger(__name__)

    def research_brainstorm(self, topic: str) -> str:
        """Generate research ideas and suggestions based on the knowledge base and papers."""
        try:
            # Get relevant documents from both code and papers
            relevant_docs = self.rag.query(topic, top_k=5)
            
            # Create a research brainstorming prompt
            prompt = f"""As a bioinformatics research assistant, help brainstorm ideas for the following topic.
            
Topic: {topic}

Relevant Context from Papers and Code:
{relevant_docs}

Please provide a comprehensive research plan that:

1. Literature Review & Current State:
   - Summarize key findings from the relevant papers
   - Identify gaps in current research
   - Highlight important methodologies used

2. Research Questions:
   - Formulate specific questions based on the literature
   - Identify areas that need further investigation
   - Suggest novel angles based on the papers

3. Methodology Suggestions:
   - Recommend approaches based on successful methods in the papers
   - Suggest improvements or modifications to existing methods
   - Consider computational requirements and feasibility

4. Potential Challenges:
   - Identify technical challenges mentioned in the papers
   - Suggest solutions based on existing approaches
   - Highlight areas that need innovative solutions

5. Implementation Strategy:
   - Suggest code structure based on existing implementations
   - Recommend libraries and tools used in similar research
   - Outline key components needed

6. Next Steps:
   - Prioritize research questions
   - Suggest immediate actions
   - Outline a timeline for implementation

Format your response in a clear, structured way, with specific references to the papers and code when relevant."""

            # Get response from GPT
            response = self.rag.client.chat.completions.create(
                model=self.rag.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics research expert that helps with research brainstorming and planning. You excel at connecting ideas from papers with practical implementation approaches."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error in research_brainstorm: {e}")
            raise

    def analyze_paper_connections(self, topic: str) -> str:
        """Analyze connections between papers and code for a given topic."""
        try:
            # Get relevant documents
            relevant_docs = self.rag.query(topic, top_k=5)
            
            prompt = f"""Analyze the connections between papers and code implementations for the following topic.

Topic: {topic}

Relevant Context:
{relevant_docs}

Please provide:

1. Paper-Code Connections:
   - Identify papers that have corresponding code implementations
   - Highlight how the code implements paper methodologies
   - Note any gaps between paper descriptions and implementations

2. Implementation Patterns:
   - Common libraries and tools used across implementations
   - Similar approaches in different papers
   - Unique implementation strategies

3. Integration Opportunities:
   - How different implementations could be combined
   - Potential improvements based on paper suggestions
   - Areas where new implementations are needed

Format your response to clearly show the relationships between papers and their implementations."""

            response = self.rag.client.chat.completions.create(
                model=self.rag.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics expert that analyzes connections between research papers and their implementations."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error in analyze_paper_connections: {e}")
            raise

    def get_implementation_suggestions(self, topic: str) -> str:
        """Get specific implementation suggestions based on papers and existing code."""
        try:
            relevant_docs = self.rag.query(topic, top_k=5)
            
            prompt = f"""Based on the following papers and code, provide specific implementation suggestions.

Topic: {topic}

Relevant Context:
{relevant_docs}

Please provide:

1. Code Structure:
   - Recommended project organization
   - Key modules and their responsibilities
   - Data flow and processing pipeline

2. Technology Stack:
   - Programming languages and frameworks
   - Key libraries and tools
   - Version control and deployment suggestions

3. Implementation Details:
   - Specific algorithms and methods to implement
   - Performance optimization techniques
   - Testing and validation approaches

4. Integration Points:
   - How to integrate with existing systems
   - API design suggestions
   - Data format and storage recommendations

Format your response with specific code examples and implementation details when possible."""

            response = self.rag.client.chat.completions.create(
                model=self.rag.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics software engineer that provides detailed implementation guidance."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error in get_implementation_suggestions: {e}")
            raise

    def analyze_papers(self, topic: str, top_k: int = None) -> Dict[str, Any]:
        """Analyze connections between papers on a given topic.
        
        Args:
            topic: The topic to analyze papers for
            top_k: Number of most relevant papers to analyze (optional)
            
        Returns:
            Dictionary containing:
            - papers: List of relevant papers with their metadata
            - connections: List of connections between papers
            - summary: Overall analysis summary
        """
        try:
            # Get relevant papers using the RAG system
            response = self.rag.query(topic)
            
            # Convert response to string if it's not already
            if response is None:
                response = ""
            elif not isinstance(response, str):
                response = str(response)
            
            if not response.strip():
                return {
                    "papers": [],
                    "connections": [],
                    "summary": "No relevant papers found for the given topic."
                }
            
            # Parse the response to extract paper information
            # The response should be a string containing paper information
            # We'll use GPT to extract structured information
            extraction_prompt = f"""Extract paper information from the following text. For each paper, identify:
1. Title
2. Authors
3. Year
4. Abstract
5. URL (if available)

Text:
{response}

Format the output as a JSON array of paper objects, each containing title, authors (array), year, abstract, and url fields. If a field is not available, use null or an empty string."""

            try:
                extraction_response = self.rag.client.chat.completions.create(
                    model=self.rag.model,
                    messages=[
                        {"role": "system", "content": "You are a paper information extraction expert. Extract structured information about papers from text. Always return valid JSON."},
                        {"role": "user", "content": extraction_prompt}
                    ]
                )
                
                # Parse the JSON response
                import json
                try:
                    papers_info = json.loads(extraction_response.choices[0].message.content)
                    if not isinstance(papers_info, list):
                        papers_info = []
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse JSON response from GPT")
                    papers_info = []
                
                if not papers_info:
                    return {
                        "papers": [],
                        "connections": [],
                        "summary": "No paper information could be extracted from the response."
                    }
                
            except Exception as e:
                self.logger.error(f"Failed to extract paper information: {str(e)}")
                return {
                    "papers": [],
                    "connections": [],
                    "summary": f"Failed to extract paper information: {str(e)}"
                }
            
            # Analyze connections between papers
            connections = []
            for i in range(len(papers_info)):
                for j in range(i + 1, len(papers_info)):
                    paper1 = papers_info[i]
                    paper2 = papers_info[j]
                    
                    # Ensure paper objects are dictionaries
                    if not isinstance(paper1, dict):
                        paper1 = {"title": str(paper1)}
                    if not isinstance(paper2, dict):
                        paper2 = {"title": str(paper2)}
                    
                    # Create prompt for connection analysis
                    prompt = f"""Analyze the connection between these two papers:

Paper 1: {paper1.get('title', 'Unknown')} ({paper1.get('year', 'Unknown')})
Abstract: {paper1.get('abstract', 'No abstract available')}

Paper 2: {paper2.get('title', 'Unknown')} ({paper2.get('year', 'Unknown')})
Abstract: {paper2.get('abstract', 'No abstract available')}

Please identify:
1. Common themes or methodologies
2. How the papers complement or build upon each other
3. Key differences in approach or findings
4. Potential impact on the field

Provide a concise analysis (2-3 paragraphs)."""

                    try:
                        response = self.rag.client.chat.completions.create(
                            model=self.rag.model,
                            messages=[
                                {"role": "system", "content": "You are a research paper analysis expert. Provide clear, concise analysis of connections between papers."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        connection_analysis = response.choices[0].message.content
                        
                        connections.append({
                            "paper1": paper1.get("title", "Unknown"),
                            "paper2": paper2.get("title", "Unknown"),
                            "analysis": connection_analysis
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to analyze connection between papers: {str(e)}")
                        continue
            
            # Generate overall summary
            summary_prompt = f"""Based on the following papers and their connections, provide a comprehensive summary of the current state of research on {topic}:

Papers:
{chr(10).join([f"- {p.get('title', 'Unknown')} ({p.get('year', 'Unknown')})" for p in papers_info])}

Connections:
{chr(10).join([f"- {c['paper1']} ↔ {c['paper2']}: {c['analysis'][:200]}..." for c in connections])}

Please provide:
1. Key themes and trends
2. Major findings and contributions
3. Gaps in current research
4. Future directions

Provide a concise summary (3-4 paragraphs)."""

            try:
                response = self.rag.client.chat.completions.create(
                    model=self.rag.model,
                    messages=[
                        {"role": "system", "content": "You are a research synthesis expert. Provide clear, comprehensive summaries of research topics."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                summary = response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Failed to generate summary: {str(e)}")
                summary = "Failed to generate comprehensive summary."
            
            return {
                "papers": papers_info,
                "connections": connections,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze papers: {str(e)}")
            raise RuntimeError(f"Failed to analyze papers: {str(e)}") 