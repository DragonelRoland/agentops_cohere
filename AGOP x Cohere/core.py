from typing import Dict, Any, List, Optional
import logging
from enum import Enum
import os
from dotenv import load_dotenv
from cohere import Client as CohereClient
import sys
from prompts import generate_prompt, PROMPT_TYPES
import agentops
import networkx as nx
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt

load_dotenv()
agentops.init(os.getenv('AGENTOPS_API_KEY'))

cohere_api_key = os.getenv('COHERE_API_KEY')

print(f"COHERE_API_KEY: {cohere_api_key}")

class NodeStatus(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3

class ExecutionError(Exception):
    pass

class Node:
    def __init__(self, node_id: str, node_type: str):
        self.id = node_id
        self.type = node_type
        self.data: Dict[str, Any] = {}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Process method must be implemented by subclasses")

class LLMNode(Node):
    def __init__(self, node_id: str):
        super().__init__(node_id, "LLM")
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        self.client = CohereClient(api_key=api_key)

    def call_llm(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        try:
            response = self.client.generate(
                model='command',
                prompt=f"{system_message}\n\n{prompt}",
                max_tokens=300,
                temperature=0.7,
            )
            return response.generations[0].text.strip()
        except Exception as e:
            logging.error(f"Cohere API error: {str(e)}")
            raise

class UINode(LLMNode):
    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing UI Node")
        print("Starting user input collection...")
        user_input = {}
        
        user_input['research_query'] = input("Enter your research query: ")
        if user_input['research_query'].lower() in ['exit', 'stop']:
            print("Exiting the program.")
            sys.exit(0)
        
        research_types = ['products', 'companies', 'market', 'investors']
        print("Select research type:")
        for i, t in enumerate(research_types, 1):
            print(f"{i}. {t}")
        research_type = int(input("Enter the number of your choice: "))
        user_input['research_type'] = research_types[research_type - 1]
        
        agent_types = ['CEO', 'investor', 'expert']
        print("Select agent type:")
        for i, t in enumerate(agent_types, 1):
            print(f"{i}. {t}")
        agent_type = int(input("Enter the number of your choice: "))
        user_input['agent_type'] = agent_types[agent_type - 1]
        
        output_types = ['table', 'PDF', 'text']
        print("Select output type:")
        for i, t in enumerate(output_types, 1):
            print(f"{i}. {t}")
        output_type = int(input("Enter the number of your choice: "))
        user_input['output_type'] = output_types[output_type - 1]
        
        user_input['job_role'] = input("Enter your job role: ")
        user_input['company'] = input("Enter your company name: ")
        user_input['deliverable'] = input("Enter the expected deliverable: ")
        
        context = f"Research query: {user_input['research_query']}\n" \
                  f"Research type: {user_input['research_type']}\n" \
                  f"Agent type: {user_input['agent_type']}\n" \
                  f"Output type: {user_input['output_type']}\n" \
                  f"Job role: {user_input['job_role']}\n" \
                  f"Company: {user_input['company']}\n" \
                  f"Deliverable: {user_input['deliverable']}"
        
        prompt = generate_prompt(context, 'ui_vector')
        vector = self.call_llm(prompt)
        
        return {"user_input": user_input, "vector": vector, "context": context}

class ScraperNode(LLMNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)

    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Scraper Node")
        
        vector = input_data.get("vector", "")
        context = input_data.get("context", "")
        
        # Placeholder for scraping process
        print("Scraping process placeholder")
        mock_scraped_data = "This is mock scraped data for demonstration purposes."
        
        prompt = generate_prompt(f"{context}\n\nSearch results: {mock_scraped_data}", 'scraper_analysis')
        analysis = self.call_llm(prompt)
        
        return {"scraped_data": mock_scraped_data, "relevance_analysis": analysis}

class AnalysisNode(LLMNode):
    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Analysis Node")
        data = input_data.get('scraped_data', '')
        context = input_data.get("context", "")
        
        prompt = generate_prompt(f"{context}\n\nData to analyze: {data}", 'analysis_options')
        options = self.call_llm(prompt)
        
        return {"options": options}

class AnalysisNode2(LLMNode):
    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Analysis Node 2")
        data = input_data.get('scraped_data', '')
        context = input_data.get("context", "")
        
        prompt = generate_prompt(f"{context}\n\nData to analyze: {data}", 'analysis_key_info')
        key_info = self.call_llm(prompt)
        
        prompt_options = generate_prompt(f"{context}\n\nKey information: {key_info}", 'analysis_presentation_options')
        options = self.call_llm(prompt_options)
        
        return {"key_information": key_info, "options": options}

class AnalysisFeedbackLoop(LLMNode):
    def __init__(self, node_id: str, max_iterations: int = 1):
        super().__init__(node_id)
        self.max_iterations = max_iterations

    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Analysis Feedback Loop")
        current_data = input_data
        
        for _ in range(self.max_iterations):
            # Analysis step
            analysis_result = self.analyze(current_data)
            
            # Feedback step
            feedback_result = self.get_feedback(analysis_result)
            
            current_data = feedback_result
        
        return current_data

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Analyze the following data and provide 3 options:\n{data}"
        options = self.call_llm(prompt)
        return {"options": options}

    def get_feedback(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        options = analysis_result.get('options', '')
        print("\nAnalysis Options:")
        print(options)
        choice = input("Select the best option (1, 2, or 3): ")
        rating = input("Rate the quality of the analysis from 1-10: ")
        
        feedback = f"Option {choice} was selected. Quality rating: {rating}/10"
        
        return {
            "feedback": feedback,
            "selected_option": int(choice),
            "feedback_score": int(rating)
        }

class OutputNode(LLMNode):
    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Output Node")
        key_info = input_data.get('key_information', '')
        output_style = input_data.get('output_type', 'text')
        context = input_data.get("context", "")
        
        prompt = generate_prompt(f"{context}\n\nOutput style: {output_style}\n\nKey information: {key_info}", 'output_generation')
        document = self.call_llm(prompt)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{timestamp}"

        # Create output based on the specified style
        if output_style == 'table':
            filename += ".csv"
            self.create_csv_output(document, filename)
        elif output_style == 'PDF':
            filename += ".txt"  # Placeholder for PDF (requires additional libraries)
            self.create_text_output(document, filename)
        else:  # Default to text
            filename += ".txt"
            self.create_text_output(document, filename)
        
        return {"document": document, "output_type": output_style, "output_file": filename}

    def create_csv_output(self, data, filename):
        # Assuming the data is a string that can be split into rows and columns
        rows = [row.split(',') for row in data.split('\n')]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def create_text_output(self, data, filename):
        with open(filename, 'w') as file:
            file.write(data)

class FinalFeedbackNode(LLMNode):
    def process(self, input_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        if session:
            print("Processing Final Feedback Node")
        document = input_data.get('document', '')
        context = input_data.get("context", "")
        
        prompt = generate_prompt(f"{context}\n\nFinal document: {document}", 'final_feedback')
        feedback = self.call_llm(prompt)
        
        rating = input("Rate the overall quality from 1-10: ")
        
        return {
            "final_feedback": feedback,
            "final_rating": int(rating)
        }

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}
        self.node_status: Dict[str, NodeStatus] = {}
        self.research_vector: Optional[str] = None
        self.context: Optional[str] = None
        self.logger = logging.getLogger(__name__)

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.edges[node.id] = []
        self.node_status[node.id] = NodeStatus.NOT_STARTED
        self.logger.info(f"Added node: {node.id} of type {node.type}")

    def add_edge(self, from_node_id: str, to_node_id: str):
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
        self.edges[from_node_id].append(to_node_id)
        self.logger.info(f"Added edge: {from_node_id} -> {to_node_id}")

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_node_status(self, node_id: str) -> NodeStatus:
        return self.node_status.get(node_id, NodeStatus.NOT_STARTED)

    def reset_execution_status(self):
        for node_id in self.nodes:
            self.node_status[node_id] = NodeStatus.NOT_STARTED

    def execute(self, initial_data: Dict[str, Any], start_node_id: str = "ui", session=None) -> Dict[str, Any]:
        print(f"Starting execution with node: {start_node_id}")
        result = initial_data
        visited = set()
        stack = [start_node_id]

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)

            print(f"Processing node: {node_id}")
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found in graph")

            try:
                result = node.process(result, session)
                if "context" in result and self.context is None:
                    self.set_context(result["context"])
                print(f"Node {node_id} processed successfully")
            except SystemExit:
                print("Program stopped by user.")
                return result
            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
                if session:
                    print(f"Error: {str(e)}")
                return result

            for neighbor in self.edges.get(node_id, []):
                stack.append(neighbor)

        return result

    def topological_sort(self) -> List[str]:
        visited = set()
        stack = []

        def dfs(node_id):
            visited.add(node_id)
            for neighbor in self.edges[node_id]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)

        return list(reversed(stack))

    def execute_topological(self, initial_data: Dict[str, Any], session=None) -> Dict[str, Any]:
        self.reset_execution_status()
        result = initial_data
        execution_order = self.topological_sort()

        for node_id in execution_order:
            try:
                node = self.nodes[node_id]
                self.node_status[node_id] = NodeStatus.IN_PROGRESS
                self.logger.info(f"Starting execution of node: {node_id}")
                result = node.process(result, session)
                self.node_status[node_id] = NodeStatus.COMPLETED
                self.logger.info(f"Completed execution of node: {node_id}")
            except Exception as e:
                self.node_status[node_id] = NodeStatus.FAILED
                self.logger.error(f"Execution failed at node {node_id}: {str(e)}")
                if session:
                    print(f"Error: {str(e)}")
                raise ExecutionError(f"Execution failed at node {node_id}: {str(e)}")

        return result

    def set_research_vector(self, vector: str):
        self.research_vector = vector

    def set_context(self, context: str):
        self.context = context

    def visualize(self, output_path: str):
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, type=node.type)
        
        # Add edges
        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                G.add_edge(from_node, to_node)
        
        # Set up the plot
        plt.figure(figsize=(16, 12))  # Increase figure size
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, {node: f"{node}\n({data['type']})" for node, data in G.nodes(data=True)})
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        # Add research vector to the plot
        if self.research_vector:
            plt.text(0.5, 1.05, f"Research Vector: {self.research_vector}", 
                     horizontalalignment='center', transform=plt.gca().transAxes)
        
        # Save the graph
        plt.title("Research Graph")
        plt.axis('off')
        plt.tight_layout(pad=4.0)  # Increase padding
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

__all__ = ['Graph', 'LLMNode', 'UINode', 'ScraperNode', 'AnalysisNode', 'AnalysisNode2', 'AnalysisFeedbackLoop', 'OutputNode', 'FinalFeedbackNode', 'ExecutionError']

graph = Graph()

ui_node = UINode("ui")
scraper_node = ScraperNode("scraper")
analysis_node = AnalysisNode("analysis")
analysis_node2 = AnalysisNode2("analysis2")
analysis_feedback_node = AnalysisFeedbackLoop("analysis_feedback", max_iterations=1)
output_node = OutputNode("output")
final_feedback_node = FinalFeedbackNode("final_feedback")

graph.add_node(ui_node)
graph.add_node(scraper_node)
graph.add_node(analysis_node)
graph.add_node(analysis_node2)
graph.add_node(analysis_feedback_node)
graph.add_node(output_node)
graph.add_node(final_feedback_node)

graph.add_edge("ui", "scraper")
graph.add_edge("scraper", "analysis")
graph.add_edge("analysis", "analysis2")
graph.add_edge("analysis2", "analysis_feedback")
graph.add_edge("analysis_feedback", "output")
graph.add_edge("output", "final_feedback")

def main():
    while True:
        print("\n--- Research Assistant ---")
        choice = input("Enter 'start' to begin a new search or 'exit' to quit: ").lower()
        
        if choice == 'exit':
            print("Exiting program. Goodbye!")
            break
        elif choice == 'start':
            try:
                session = agentops.start_session()
            except Exception as e:
                print(f"Failed to start AgentOps session: {e}")
                session = None

            # Execute the graph
            initial_data = {}
            result = graph.execute(initial_data, start_node_id="ui", session=session)

            # Set the research vector and context in the graph
            if "vector" in result:
                graph.set_research_vector(result["vector"])
            if "context" in result:
                graph.set_context(result["context"])

            # Generate the graph visualization
            graph.visualize("research_graph.png")

            if "output_file" in result:
                print(f"Research output has been saved to: {result['output_file']}")
                # Optionally, display a summary or the first few lines of the output
                with open(result['output_file'], 'r') as file:
                    print("\nOutput Preview:")
                    print(file.read()[:500] + "...")  # Display first 500 characters
            else:
                print("No output file was generated.")

            print("Search completed. Graph visualization saved as 'research_graph.png'.")
            if session:
                session.end_session('Success')
            else:
                print("AgentOps session was not started, skipping end_session.")
        else:
            print("Invalid input. Please enter 'start' or 'exit'.")

if __name__ == "__main__":
    main()