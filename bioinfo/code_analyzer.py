import os
import ast
import logging
from typing import List, Dict, Any, Optional
from core.base_rag import BaseRAG

class CodeAnalyzer:
    def __init__(self, rag: BaseRAG):
        self.rag = rag
        self.logger = logging.getLogger(__name__)
        self.code_documents = []
        self.code_embeddings = []
        self.code_metadata = []

    def process_code_file(self, file_path: str) -> Dict[str, Any]:
        """Process a Python code file and extract relevant information."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                
            # Parse the code using AST
            tree = ast.parse(content)
            
            # Extract functions, classes, and their docstrings
            code_info = {
                'filename': os.path.basename(file_path),
                'functions': [],
                'classes': [],
                'imports': [],
                'dependencies': set()
            }
            
            # Extract imports and dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        code_info['imports'].append(n.name)
                        code_info['dependencies'].add(n.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        code_info['imports'].append(f"{node.module}.{node.names[0].name}")
                        code_info['dependencies'].add(node.module.split('.')[0])
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args],
                        'code': ast.unparse(node),
                        'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                    }
                    code_info['functions'].append(func_info)
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                        'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                    }
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item),
                                'args': [arg.arg for arg in item.args.args],
                                'code': ast.unparse(item),
                                'decorators': [d.id for d in item.decorator_list if isinstance(d, ast.Name)]
                            }
                            class_info['methods'].append(method_info)
                    code_info['classes'].append(class_info)
            
            return code_info
            
        except Exception as e:
            self.logger.error(f"Error processing code file {file_path}: {e}")
            raise

    def add_code_file(self, file_path: str) -> None:
        """Add a code file to the RAG system."""
        if not file_path.endswith('.py'):
            self.logger.warning(f"Skipping non-Python file: {file_path}")
            return
            
        self.logger.info(f"Processing code file: {file_path}")
        try:
            code_info = self.process_code_file(file_path)
            
            # Create documents for each function and class
            for func in code_info['functions']:
                doc = self._create_function_doc(func, code_info)
                self._add_document(doc, 'function', func['name'], code_info['filename'])
            
            for cls in code_info['classes']:
                doc = self._create_class_doc(cls, code_info)
                self._add_document(doc, 'class', cls['name'], code_info['filename'])
                
                # Add individual methods
                for method in cls['methods']:
                    method_doc = self._create_method_doc(method, cls, code_info)
                    self._add_document(method_doc, 'method', method['name'], code_info['filename'], cls['name'])
            
            # Add file-level documentation
            file_doc = self._create_file_doc(code_info)
            self._add_document(file_doc, 'file', code_info['filename'], code_info['filename'])
            
        except Exception as e:
            self.logger.error(f"Error adding code file {file_path}: {e}")
            raise

    def _create_function_doc(self, func: Dict[str, Any], code_info: Dict[str, Any]) -> str:
        """Create documentation for a function."""
        return f"""Function: {func['name']}
File: {code_info['filename']}
Arguments: {', '.join(func['args'])}
Decorators: {', '.join(func['decorators']) if func['decorators'] else 'None'}
Docstring: {func['docstring'] or 'No docstring'}
Code:
{func['code']}"""

    def _create_class_doc(self, cls: Dict[str, Any], code_info: Dict[str, Any]) -> str:
        """Create documentation for a class."""
        return f"""Class: {cls['name']}
File: {code_info['filename']}
Bases: {', '.join(cls['bases']) if cls['bases'] else 'None'}
Decorators: {', '.join(cls['decorators']) if cls['decorators'] else 'None'}
Docstring: {cls['docstring'] or 'No docstring'}
Methods: {', '.join(m['name'] for m in cls['methods'])}"""

    def _create_method_doc(self, method: Dict[str, Any], cls: Dict[str, Any], code_info: Dict[str, Any]) -> str:
        """Create documentation for a method."""
        return f"""Method: {method['name']}
Class: {cls['name']}
File: {code_info['filename']}
Arguments: {', '.join(method['args'])}
Decorators: {', '.join(method['decorators']) if method['decorators'] else 'None'}
Docstring: {method['docstring'] or 'No docstring'}
Code:
{method['code']}"""

    def _create_file_doc(self, code_info: Dict[str, Any]) -> str:
        """Create documentation for a file."""
        return f"""File: {code_info['filename']}
Imports: {', '.join(code_info['imports'])}
Dependencies: {', '.join(code_info['dependencies'])}
Functions: {', '.join(f['name'] for f in code_info['functions'])}
Classes: {', '.join(c['name'] for c in code_info['classes'])}"""

    def _add_document(self, doc: str, doc_type: str, name: str, file: str, class_name: Optional[str] = None) -> None:
        """Add a document to the RAG system."""
        try:
            embedding = self.rag.get_embedding(doc)
            self.code_documents.append(doc)
            self.code_embeddings.append(embedding)
            metadata = {
                'type': doc_type,
                'name': name,
                'file': file
            }
            if class_name:
                metadata['class'] = class_name
            self.code_metadata.append(metadata)
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            raise

    def query_code(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system specifically for code-related questions."""
        try:
            # Get embedding for the question
            question_embedding = self.rag.get_embedding(question)
            
            # Calculate similarities with code documents
            similarities = self.rag.cosine_similarity(
                [question_embedding],
                self.code_embeddings
            )[0]
            
            # Get top k most similar documents
            top_indices = self.rag.np.argsort(similarities)[-top_k:][::-1]
            relevant_docs = [self.code_documents[i] for i in top_indices]
            relevant_metadata = [self.code_metadata[i] for i in top_indices]
            
            # Create prompt with relevant documents
            prompt = f"""You are a bioinformatics expert. Based on the following code context, please answer the question.
            
Context:
{' '.join(relevant_docs)}

Question: {question}

Please provide a detailed answer that:
1. Explains the code functionality
2. Highlights important implementation details
3. Suggests potential improvements or alternatives
4. References specific parts of the code when relevant

Answer:"""
            
            # Get response from GPT
            response = self.rag.client.chat.completions.create(
                model=self.rag.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics expert that answers questions based on the provided code context."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error querying code: {e}")
            raise

def find_code_files(directory: str) -> List[str]:
    """Recursively find all Python files in directory and its subdirectories."""
    code_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                code_files.append(os.path.join(root, file))
    return code_files 