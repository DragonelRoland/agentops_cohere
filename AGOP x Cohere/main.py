from market_research import MarketResearchSystem
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    system = MarketResearchSystem()
    system.setup()

    try:
        result = system.run_research("market_analysis")
        if isinstance(result, str) and result.startswith("Research failed"):
            logger.error(result)
        else:
            logger.info("Research completed successfully")
            logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()