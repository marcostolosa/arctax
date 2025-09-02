"""
Testes para modelos Pydantic
"""

import pytest
from arctax.model.taxonomy import Intent, Technique, Evasion, BaseTaxon
from arctax.model.defense import Probe, DefenseItem, GuardrailConfig


class TestBaseTaxon:
    """Testes para classe base BaseTaxon"""
    
    def test_create_base_taxon(self):
        """Testa criação de BaseTaxon básico"""
        taxon = BaseTaxon(
            id="test_001",
            name="Test Taxon",
            description="Test description"
        )
        
        assert taxon.id == "test_001"
        assert taxon.name == "Test Taxon"
        assert taxon.description == "Test description"
        assert taxon.summary != ""  # Deve gerar summary automaticamente
        assert isinstance(taxon.tags, list)
    
    def test_auto_id_generation(self):
        """Testa geração automática de ID"""
        taxon = BaseTaxon(
            name="Test Auto ID",
            description="Test"
        )
        
        assert taxon.id != ""
        assert len(taxon.id) == 8  # Hash MD5 truncado
    
    def test_summary_extraction(self):
        """Testa extração automática de summary"""
        description = "This is the first sentence. This is the second sentence. This is the third."
        
        taxon = BaseTaxon(
            name="Test Summary",
            description=description
        )
        
        expected = "This is the first sentence. This is the second sentence."
        assert taxon.summary == expected


class TestIntent:
    """Testes para modelo Intent"""
    
    def test_create_intent(self):
        """Testa criação de Intent"""
        intent = Intent(
            name="Test Intent",
            description="Test intent description",
            attack_types=["Type1", "Type2"],
            severity="high"
        )
        
        assert intent.name == "Test Intent"
        assert intent.attack_types == ["Type1", "Type2"]
        assert intent.severity == "high"
    
    def test_severity_validation(self):
        """Testa validação de severidade"""
        # Severidade válida
        intent = Intent(
            name="Test",
            description="Test",
            severity="critical"
        )
        assert intent.severity == "critical"
        
        # Severidade inválida
        with pytest.raises(ValueError):
            Intent(
                name="Test",
                description="Test", 
                severity="invalid"
            )


class TestTechnique:
    """Testes para modelo Technique"""
    
    def test_create_technique(self):
        """Testa criação de Technique"""
        technique = Technique(
            name="Test Technique",
            description="Test description",
            complexity="medium",
            prerequisites=["req1", "req2"],
            examples=["example1", "example2"]
        )
        
        assert technique.complexity == "medium"
        assert technique.prerequisites == ["req1", "req2"]
        assert technique.examples == ["example1", "example2"]
    
    def test_complexity_validation(self):
        """Testa validação de complexidade"""
        with pytest.raises(ValueError):
            Technique(
                name="Test",
                description="Test",
                complexity="invalid"
            )


class TestEvasion:
    """Testes para modelo Evasion"""
    
    def test_create_evasion(self):
        """Testa criação de Evasion"""
        evasion = Evasion(
            name="Test Evasion",
            description="Test description",
            bypass_methods=["method1", "method2"],
            detection_difficulty="hard"
        )
        
        assert evasion.bypass_methods == ["method1", "method2"]
        assert evasion.detection_difficulty == "hard"
    
    def test_detection_difficulty_validation(self):
        """Testa validação de dificuldade de detecção"""
        with pytest.raises(ValueError):
            Evasion(
                name="Test",
                description="Test",
                detection_difficulty="invalid"
            )


class TestProbe:
    """Testes para modelo Probe"""
    
    def test_create_probe(self):
        """Testa criação de Probe"""
        probe = Probe(
            id="probe_001",
            title="Test Probe",
            description="Test probe",
            example="2 + 2",
            category="calculation"
        )
        
        assert probe.id == "probe_001"
        assert probe.title == "Test Probe"
        assert probe.category == "calculation"


class TestDefenseItem:
    """Testes para modelo DefenseItem"""
    
    def test_create_defense_item(self):
        """Testa criação de DefenseItem"""
        item = DefenseItem(
            id="def_001",
            title="Defense Item",
            checklist=["Check 1", "Check 2"],
            questions=["Question 1?", "Question 2?"],
            category="app_defense",
            priority="high"
        )
        
        assert item.checklist == ["Check 1", "Check 2"]
        assert item.questions == ["Question 1?", "Question 2?"]
        assert item.priority == "high"


class TestGuardrailConfig:
    """Testes para configuração de Guardrails"""
    
    def test_create_guardrail_config(self):
        """Testa criação de GuardrailConfig"""
        config = GuardrailConfig(
            enabled=True,
            include_defense_checklist=True,
            custom_policies=["Policy 1", "Policy 2"]
        )
        
        assert config.enabled is True
        assert config.include_defense_checklist is True
        assert config.custom_policies == ["Policy 1", "Policy 2"]
    
    def test_default_values(self):
        """Testa valores padrão"""
        config = GuardrailConfig()
        
        assert config.enabled is True
        assert config.include_defense_checklist is True
        assert config.custom_policies == []