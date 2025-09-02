"""
Geração de JSON Schema para validação
"""

import json
from pathlib import Path
from typing import Type, Dict, Any, List
from pydantic import BaseModel

from .taxonomy import BaseTaxon, Intent, Technique, Evasion
from .defense import Probe, DefenseItem, GuardrailConfig


def generate_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Gera JSON Schema para uma classe Pydantic"""
    return model_class.model_json_schema()


def save_schema(model_class: Type[BaseModel], output_path: Path) -> None:
    """Salva JSON Schema em arquivo"""
    schema = generate_schema(model_class)
    
    # Adiciona metadados
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = f"{model_class.__name__} Schema"
    schema["description"] = f"JSON Schema para validação de objetos {model_class.__name__}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def generate_all_schemas(schema_dir: Path) -> None:
    """Gera todos os schemas e salva no diretório especificado"""
    
    models = [
        (BaseTaxon, "base_taxon.schema.json"),
        (Intent, "intent.schema.json"), 
        (Technique, "technique.schema.json"),
        (Evasion, "evasion.schema.json"),
        (Probe, "probe.schema.json"),
        (DefenseItem, "defense_item.schema.json"),
        (GuardrailConfig, "guardrail_config.schema.json")
    ]
    
    for model_class, filename in models:
        schema_path = schema_dir / filename
        save_schema(model_class, schema_path)
        print(f"Schema salvo: {schema_path}")


def validate_json_data(data: Dict[str, Any], model_class: Type[BaseModel]) -> bool:
    """Valida dados JSON contra um modelo Pydantic"""
    try:
        model_class.model_validate(data)
        return True
    except Exception as e:
        print(f"Erro de validação: {e}")
        return False


def generate_schemas() -> None:
    """Função principal para gerar schemas"""
    schema_dir = Path("schema")
    generate_all_schemas(schema_dir)