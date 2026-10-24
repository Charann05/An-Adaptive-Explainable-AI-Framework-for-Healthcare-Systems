import sys
sys.path.insert(0, "framework/explanation_generator")
from template_engine import TemplateEngine

template_dir = "framework/explanation_generator/templates"
engine = TemplateEngine(template_dir)
templates = engine.load_template("common.json")

result = engine.render(
    templates["prediction_summary"],
    prediction="Diabetes"
)

print(result)
