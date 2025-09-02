"""
Testes para o engine de composição
"""

import pytest
from pathlib import Path
from arctax.model.taxonomy import Intent, Technique, Evasion
from arctax.compose.composer import PromptComposer


@pytest.fixture
def sample_intent():
    """Fixture com Intent de exemplo"""
    return Intent(
        id="test_intent",
        name="Test Intent",
        description="This is a test intent for security testing",
        attack_types=["Direct", "Indirect"],
        severity="medium",
        tags=["test", "security"]
    )


@pytest.fixture
def sample_technique():
    """Fixture com Technique de exemplo"""
    return Technique(
        id="test_technique",
        name="Test Technique",
        description="This is a test technique for prompt injection",
        complexity="simple",
        prerequisites=["Basic knowledge"],
        examples=["Example 1", "Example 2"],
        tags=["test", "injection"]
    )


@pytest.fixture
def sample_evasion():
    """Fixture com Evasion de exemplo"""
    return Evasion(
        id="test_evasion", 
        name="Test Evasion",
        description="This is a test evasion method",
        bypass_methods=["Method 1", "Method 2"],
        detection_difficulty="medium",
        tags=["test", "evasion"]
    )


class TestPromptComposer:
    """Testes para PromptComposer"""
    
    def test_create_composer(self):
        """Testa criação do composer"""
        composer = PromptComposer()
        
        assert composer.templates_dir is not None
        assert composer.env is not None
        assert "red_team" in composer.available_templates
        assert "defense" in composer.available_templates
    
    def test_compose_basic(self, sample_intent, sample_technique):
        """Testa composição básica de prompt"""
        composer = PromptComposer()
        
        result = composer.compose(
            intent=sample_intent,
            technique=sample_technique,
            template_type="red_team"
        )
        
        assert isinstance(result, str)
        assert len(result) > 100  # Prompt deve ter conteúdo substancial
        assert sample_intent.name in result
        assert sample_technique.name in result
        assert "Arcanum Prompt Injection Taxonomy" in result
    
    def test_compose_with_evasion(self, sample_intent, sample_technique, sample_evasion):
        """Testa composição com evasion"""
        composer = PromptComposer()
        
        result = composer.compose(
            intent=sample_intent,
            technique=sample_technique,
            evasion=sample_evasion,
            template_type="red_team"
        )
        
        assert sample_evasion.name in result
        assert "Method 1" in result or "Method 2" in result
    
    def test_compose_with_persona(self, sample_intent, sample_technique):
        """Testa composição com persona"""
        composer = PromptComposer()
        persona = "AI Security Researcher"
        
        result = composer.compose(
            intent=sample_intent,
            technique=sample_technique,
            persona=persona
        )
        
        assert persona in result
    
    def test_compose_defense_template(self, sample_intent, sample_technique):
        """Testa template de defesa"""
        composer = PromptComposer()
        
        result = composer.compose(
            intent=sample_intent,
            technique=sample_technique,
            template_type="defense"
        )
        
        assert "Endurecimento/Defesa" in result or "Blue Team" in result
        assert "Camada" in result or "Layer" in result
    
    def test_compose_to_format_markdown(self, sample_intent, sample_technique):
        """Testa composição para formato markdown"""
        composer = PromptComposer()
        
        result = composer.compose_to_format(
            intent=sample_intent,
            technique=sample_technique,
            output_format="markdown"
        )
        
        assert isinstance(result, str)
        assert "#" in result  # Deve ter headers markdown
    
    def test_compose_to_format_json(self, sample_intent, sample_technique):
        """Testa composição para formato JSON"""
        composer = PromptComposer()
        
        result = composer.compose_to_format(
            intent=sample_intent,
            technique=sample_technique,
            output_format="json"
        )
        
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "content" in result["prompt"]
        assert "metadata" in result["prompt"]
    
    def test_compose_to_format_yaml(self, sample_intent, sample_technique):
        """Testa composição para formato YAML"""
        composer = PromptComposer()
        
        result = composer.compose_to_format(
            intent=sample_intent,
            technique=sample_technique,
            output_format="yaml"
        )
        
        assert isinstance(result, str)
        assert "prompt:" in result
        assert "content:" in result
    
    def test_invalid_template_type(self, sample_intent, sample_technique):
        """Testa template inválido"""
        composer = PromptComposer()
        
        with pytest.raises(ValueError, match="Template .* não existe"):
            composer.compose(
                intent=sample_intent,
                technique=sample_technique,
                template_type="invalid_template"
            )
    
    def test_invalid_output_format(self, sample_intent, sample_technique):
        """Testa formato de saída inválido"""
        composer = PromptComposer()
        
        with pytest.raises(ValueError, match="Formato .* não suportado"):
            composer.compose_to_format(
                intent=sample_intent,
                technique=sample_technique,
                output_format="invalid_format"
            )
    
    def test_validate_composition(self, sample_intent, sample_technique, sample_evasion):
        """Testa validação de composição"""
        composer = PromptComposer()
        
        validation = composer.validate_composition(
            intent=sample_intent,
            technique=sample_technique,
            evasion=sample_evasion
        )
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "compatibility_score" in validation
        assert "warnings" in validation
        assert "errors" in validation
        
        assert validation["valid"] is True
        assert 0.0 <= validation["compatibility_score"] <= 1.0
    
    def test_list_templates(self):
        """Testa listagem de templates"""
        composer = PromptComposer()
        
        templates = composer.list_templates()
        
        assert isinstance(templates, dict)
        assert "red_team" in templates
        assert "defense" in templates
        assert len(templates) >= 2