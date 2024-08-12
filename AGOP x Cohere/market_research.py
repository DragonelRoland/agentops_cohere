from core import Graph, UINode, ScraperNode, AnalysisNode, AnalysisNode2, AnalysisFeedbackLoop, OutputNode, FinalFeedbackNode, ExecutionError
from trust import TrustVerifier
from workflow import UIWorkflow
from output import OutputGenerator
from typing import Any, Dict

class MarketResearchSystem:
    def __init__(self):
        self.graph = Graph()
        self.trust_verifier = TrustVerifier()
        self.ui_workflow = UIWorkflow()
        self.output_generator = OutputGenerator()

    def setup(self):
        # Create nodes
        data_collection = ScraperNode("data_collection")
        analysis = AnalysisNode("analysis")
        expert_consultation = AnalysisNode2("expert_consultation")
        report_generation = OutputNode("report_generation")

        # Add nodes to graph
        self.graph.add_node(data_collection)
        self.graph.add_node(analysis)
        self.graph.add_node(expert_consultation)
        self.graph.add_node(report_generation)

        # Define edges
        self.graph.add_edge("data_collection", "analysis")
        self.graph.add_edge("analysis", "expert_consultation")
        self.graph.add_edge("expert_consultation", "report_generation")

    def run_research(self, query: str) -> Dict[str, Any]:
        initial_data = {"query": query}
        result = self.graph.execute("data_collection", initial_data)
        return result

__all__ = ['MarketResearchSystem']