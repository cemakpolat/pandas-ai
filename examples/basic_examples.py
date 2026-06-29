"""
Example scenarios for pandas-ai analysis.
"""

import sys
import os
from typing import List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pychartai_core import DataManager, DataAnalyzer


class AnalysisExample:
    """Base class for analysis examples."""
    
    def __init__(self, name: str, description: str):
        """Initialize example."""
        self.name = name
        self.description = description
        self.data_manager = DataManager()
        self.analyzer: DataAnalyzer = None
    
    def run(
        self,
        model_name: str = "llama3.2",
        provider_type: str = "ollama",
        api_key: str = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run the example."""
        raise NotImplementedError
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print results."""
        print(f"\n{'='*60}")
        print(f"Example: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*60}\n")
        
        for key, value in results.items():
            print(f"{key}:")
            if isinstance(value, list):
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"  {value}")
            print()


class SalesAnalysisExample(AnalysisExample):
    """Sales data analysis example."""
    
    def __init__(self):
        super().__init__(
            name="Sales Data Analysis",
            description="Analyze sales data to find trends and top performers"
        )
    
    def run(
        self,
        model_name: str = "llama3.2",
        provider_type: str = "ollama",
        api_key: str = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run sales analysis."""
        if self.analyzer is None:
            self.analyzer = DataAnalyzer(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                verbose=verbose,
            )
        
        df = self.data_manager.create_sample_data("sales_data", "sales")
        
        queries = [
            "What are the top 3 best-selling products?",
            "Which salesperson has the highest total sales?",
            "What is the average sale value by region?",
            "Summarize the sales data by showing total sales per region",
        ]
        
        results = {
            "Data Shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
            "Date Range": f"{df['date'].min()} to {df['date'].max()}",
            "Analysis Results": []
        }
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query}", flush=True)
            result = self.analyzer.analyze(df, query)
            results["Analysis Results"].append(f"Q: {query}\nA: {result}")
        
        return results


class WeatherAnalysisExample(AnalysisExample):
    """Weather data analysis example."""
    
    def __init__(self):
        super().__init__(
            name="Weather Data Analysis",
            description="Analyze weather patterns and conditions"
        )
    
    def run(
        self,
        model_name: str = "llama3.2",
        provider_type: str = "ollama",
        api_key: str = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run weather analysis."""
        if self.analyzer is None:
            self.analyzer = DataAnalyzer(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                verbose=verbose,
            )
        
        df = self.data_manager.create_sample_data("weather_data", "weather")
        
        queries = [
            "What is the average temperature across all cities?",
            "Which city has the highest rainfall?",
            "Calculate average humidity by city",
            "Identify extreme weather patterns",
        ]
        
        results = {
            "Data Shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
            "Cities": df["city"].unique().tolist(),
            "Analysis Results": []
        }
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query}", flush=True)
            result = self.analyzer.analyze(df, query)
            results["Analysis Results"].append(f"Q: {query}\nA: {result}")
        
        return results


class EcommerceAnalysisExample(AnalysisExample):
    """E-commerce order data analysis."""

    def __init__(self):
        super().__init__(
            name="E-commerce Analysis",
            description="Analyse orders, discounts, ratings and returns across categories"
        )

    def run(
        self,
        model_name: str = "llama3.2",
        provider_type: str = "ollama",
        api_key: str = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if self.analyzer is None:
            self.analyzer = DataAnalyzer(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                verbose=verbose,
            )

        df = self.data_manager.create_sample_data("ecommerce_data", "ecommerce")

        queries = [
            "What is the total revenue (final_value) by category?",
            "Which country has the highest average order value?",
            "What is the return rate (returned=1) by category?",
        ]

        results = {
            "Data Shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
            "Categories": df["category"].unique().tolist(),
            "Analysis Results": [],
        }

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query}", flush=True)
            result = self.analyzer.analyze(df, query)
            results["Analysis Results"].append(f"Q: {query}\nA: {result}")

        return results


def run_all_examples(
    models: List[str] = None,
    provider_type: str = "ollama",
    api_key: str = None,
    verbose: bool = False,
) -> None:
    """Run all examples with specified models."""
    if models is None:
        models = ["llama3.2"]
    
    examples = [
        SalesAnalysisExample(),
        WeatherAnalysisExample(),
        EcommerceAnalysisExample(),
    ]
    
    model = models[0]
    print(f"Initializing model '{model}'...", flush=True)
    analyzer = DataAnalyzer(
        model_name=model,
        provider_type=provider_type,
        api_key=api_key,
        verbose=verbose,
    )
    print(f"✓ Model ready.\n", flush=True)
    
    for example in examples:
        try:
            example.analyzer = analyzer
            print(f"\n--- {example.name} ---", flush=True)
            results = example.run(
                model_name=model,
                provider_type=provider_type,
                api_key=api_key,
                verbose=verbose,
            )
            example.print_results(results)
        except Exception as e:
            print(f"Error running {example.name}: {str(e)}")
