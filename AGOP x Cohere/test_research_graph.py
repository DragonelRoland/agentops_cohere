import os
print(f"COHERE_API_KEY: {'Set' if os.getenv('COHERE_API_KEY') else 'Not set'}")

from core import Graph, UINode, ScraperNode, AnalysisFeedbackLoop, OutputNode, FinalFeedbackNode

def test_research_workflow():
    print("Starting research workflow...")
    graph = Graph()

    print("Creating nodes...")
    ui_node = UINode("ui")
    scraper_node = ScraperNode("scraper")
    analysis_feedback_node = AnalysisFeedbackLoop("analysis_feedback", max_iterations=1)
    output_node = OutputNode("output")
    final_feedback_node = FinalFeedbackNode("final_feedback")

    print("Adding nodes to graph...")
    graph.add_node(ui_node)
    graph.add_node(scraper_node)
    graph.add_node(analysis_feedback_node)
    graph.add_node(output_node)
    graph.add_node(final_feedback_node)

    print("Adding edges to graph...")
    graph.add_edge("ui", "scraper")
    graph.add_edge("scraper", "analysis_feedback")
    graph.add_edge("analysis_feedback", "output")
    graph.add_edge("output", "final_feedback")

    print("Executing graph...")
    result = graph.execute({}, start_node_id="ui")
    print("Final Result:", result)

if __name__ == "__main__":
    test_research_workflow()