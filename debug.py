# test_fixed.py
from analyzer import CardiovascularAnalyzer

analyzer = CardiovascularAnalyzer()
analyzer.load_file("data/REST.acq")  # Your file path
analyzer.analyze_all()
print(analyzer.get_summary())