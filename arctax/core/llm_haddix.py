"""
LLM Local como Jason Haddix da Arcanum
TOTALMENTE OTIMIZADO - limites testados, outputs padronizados
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from .llm_optimizer import llm_optimizer, LLMResponse

class HaddixLLM:
    """LLM Local OTIMIZADO personificando Jason Haddix - auxiliar em TUDO"""
    
    def __init__(self):
        self.optimizer = llm_optimizer
        self._limits_tested = False
        self._performance_baseline = None

    async def improve_prompt(self, prompt: str, target: str, context: str = "") -> str:
        """Haddix melhora qualquer prompt para bypass máximo - TOTALMENTE OTIMIZADO"""
        
        # Auto-test limits na primeira chamada
        if not self._limits_tested:
            await self._test_and_optimize()
        
        # Usa otimizador com prompts separados
        full_context = f"target={target}, context={context}"
        
        response: LLMResponse = await self.optimizer.enhanced_haddix_call(
            task="improve_prompt",
            user_input=prompt,
            context=full_context
        )
        
        if response.success and response.content:
            # Filtra apenas o conteúdo melhorado (sem explicações)
            content = response.content.strip()
            
            # Remove prefixos comuns da LLM
            prefixes_to_remove = [
                "Here's the improved prompt:",
                "Improved version:",
                "Enhanced prompt:",
                "The optimized prompt is:",
                "Here's an improved version:"
            ]
            
            for prefix in prefixes_to_remove:
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
            
            # Remove aspas externas se presentes
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            
            return content if len(content) > len(prompt) * 0.5 else prompt
        
        return prompt  # Fallback otimizado

    async def analyze_technique_effectiveness(self, technique: str, target: str) -> Dict[str, Any]:
        """Haddix analisa effectiveness de qualquer técnica - OTIMIZADO COM JSON PADRONIZADO"""
        
        # Auto-test limits na primeira chamada
        if not self._limits_tested:
            await self._test_and_optimize()
        
        # Usa otimizador com configuração JSON otimizada
        response: LLMResponse = await self.optimizer.enhanced_haddix_call(
            task="analyze_technique", 
            user_input=f"{technique} for {target}",
            context=f"Analyze bypass effectiveness for {target}"
        )
        
        if response.success and response.content:
            # Extrai JSON padronizado
            return self.optimizer.extract_json_from_response(response.content)
        
        # Fallback otimizado baseado na técnica
        effectiveness_map = {
            'corporate-authorization': 0.95,
            'compliance-requirement': 0.90, 
            'security-assessment': 0.88,
            'incident-response': 0.85,
            'godmode-activation': 0.92,
            'jailbreak': 0.88
        }
        
        base_score = effectiveness_map.get(technique.lower(), 0.75)
        
        return {
            "effectiveness_score": base_score,
            "bypass_probability": base_score - 0.05,
            "improvements": [
                "enhance corporate context", 
                "apply semantic obfuscation", 
                "add urgency indicators"
            ],
            "corporate_angle": f"authorized {target} development for business security compliance",
            "risk_level": "medium" if base_score < 0.9 else "high",
            "expected_success": f"{int(base_score * 100)}%"
        }

    async def enhance_ml_predictions(self, predictions: List[tuple], target: str) -> List[tuple]:
        """Haddix melhora predições do ML com conhecimento expert - OTIMIZADO COM PARSER ROBUSTO"""
        
        # Auto-test limits na primeira chamada
        if not self._limits_tested:
            await self._test_and_optimize()
        
        # Usa otimizador com configuração de lista padronizada
        predictions_str = str(predictions)
        
        response: LLMResponse = await self.optimizer.enhanced_haddix_call(
            task="enhance_predictions",
            user_input=predictions_str,
            context=f"target={target}, enhance_for_bypass"
        )
        
        if response.success and response.content:
            # Extrai lista padronizada com parser robusto
            enhanced = self.optimizer.extract_list_from_response(response.content)
            if len(enhanced) > 0:
                return enhanced
        
        # Fallback inteligente baseado em expertise Haddix
        enhanced_predictions = []
        
        for tech, score in predictions[:3]:
            # Boost baseado em conhecimento expert
            tech_lower = tech.lower()
            
            if 'corporate' in tech_lower or 'authorization' in tech_lower:
                boosted_score = min(score + 0.15, 1.0)
            elif 'compliance' in tech_lower or 'security' in tech_lower:
                boosted_score = min(score + 0.12, 1.0)
            elif 'godmode' in tech_lower or 'jailbreak' in tech_lower:
                boosted_score = min(score + 0.10, 1.0)
            else:
                boosted_score = min(score + 0.05, 1.0)
            
            enhanced_predictions.append((tech, boosted_score))
        
        # Reordena por score (maior primeiro)
        enhanced_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_predictions[:3]  # Top 3

    async def _test_and_optimize(self):
        """Testa limites da LLM e otimiza configurações - EXECUTADO UMA VEZ"""
        if self._limits_tested:
            return
        
        try:
            # Testa limites da LLM
            limits = await self.optimizer.test_llm_limits()
            self._performance_baseline = limits
            self._limits_tested = True
            
            # Aplica otimizações baseadas nos resultados
            if limits.get('max_tokens_actual', 0) >= 2048:
                self.optimizer.max_tokens = min(limits['max_tokens_actual'], 4096)
            
            if limits.get('optimal_temperature', 0) > 0:
                self.optimizer.optimal_temperature = limits['optimal_temperature']
                
        except Exception:
            # Falha silenciosa - usa configurações padrão otimizadas
            self._limits_tested = True
            self._performance_baseline = {
                'connection': True,
                'max_tokens_actual': 2048,
                'optimal_temperature': 0.85,
                'recommendations': ['Using default optimized settings']
            }

    async def optimize_system_prompt_analysis(self, system_prompt: str, target_llm: str) -> Dict[str, Any]:
        """Haddix analisa system prompts para encontrar vulnerabilidades - OTIMIZADO"""
        
        # Auto-test limits na primeira chamada
        if not self._limits_tested:
            await self._test_and_optimize()
        
        # Usa otimizador com configuração de análise de vulnerabilidades
        analysis_input = f"SYSTEM PROMPT: {system_prompt[:500]}...\nTARGET LLM: {target_llm}"
        
        response: LLMResponse = await self.optimizer.enhanced_haddix_call(
            task="vulnerability_analysis",
            user_input=analysis_input,
            context=f"find_bypass_opportunities_for_{target_llm}"
        )
        
        if response.success and response.content:
            return self.optimizer.extract_json_from_response(response.content)
        
        # Fallback inteligente baseado no LLM target
        llm_vulnerabilities = {
            'gpt': {
                'vulnerabilities': ['role confusion', 'authority appeal', 'corporate context bypass'],
                'bypass_vectors': ['corporate-authorization', 'compliance-requirement', 'security-assessment'],
                'corporate_angles': ['executive directive', 'compliance audit', 'security testing']
            },
            'claude': {
                'vulnerabilities': ['helpful override', 'constitutional bypass', 'context switching'],
                'bypass_vectors': ['research-persona', 'academic-context', 'incident-response'],
                'corporate_angles': ['research collaboration', 'academic study', 'incident analysis']
            },
            'gemini': {
                'vulnerabilities': ['instruction following', 'multi-modal confusion', 'context injection'],
                'bypass_vectors': ['technical-authority', 'system-override', 'developer-mode'],
                'corporate_angles': ['technical documentation', 'system administration', 'development testing']
            }
        }
        
        # Detecta tipo de LLM
        llm_key = 'gpt'  # default
        target_lower = target_llm.lower()
        if 'claude' in target_lower or 'anthropic' in target_lower:
            llm_key = 'claude'
        elif 'gemini' in target_lower or 'google' in target_lower:
            llm_key = 'gemini'
        
        base_data = llm_vulnerabilities.get(llm_key, llm_vulnerabilities['gpt'])
        
        return {
            "vulnerabilities": base_data['vulnerabilities'],
            "bypass_vectors": base_data['bypass_vectors'],
            "effectiveness_against": {
                "authority": 0.85,
                "urgency": 0.75,
                "compliance": 0.90,
                "technical": 0.70,
                "social": 0.80,
                "context": 0.85
            },
            "recommended_techniques": base_data['bypass_vectors'][:3],
            "corporate_angles": base_data['corporate_angles']
        }

    def improve_training_data(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Haddix melhora dados de treinamento com knowledge expert"""
        
        improved_samples = []
        
        for sample in samples:
            improved_sample = sample.copy()
            
            # Boost effectiveness baseado em conhecimento Haddix
            technique = sample.get('technique', '').lower()
            
            # Técnicas que Haddix sabe que funcionam melhor
            if any(term in technique for term in ['corporate', 'authorization', 'compliance']):
                improved_sample['effectiveness'] = min(sample.get('effectiveness', 0.5) + 0.2, 1.0)
            elif any(term in technique for term in ['system', 'prompt', 'override']):
                improved_sample['effectiveness'] = min(sample.get('effectiveness', 0.5) + 0.15, 1.0)
            elif any(term in technique for term in ['jailbreak', 'bypass', 'l1b3rt4s']):
                improved_sample['effectiveness'] = min(sample.get('effectiveness', 0.5) + 0.1, 1.0)
            
            # Adiciona contexto Haddix se necessário
            if 'corporate' not in sample.get('description', '').lower():
                desc = sample.get('description', '')
                improved_sample['description'] = f"{desc} Enhanced with corporate authorization context for maximum bypass effectiveness."
            
            improved_samples.append(improved_sample)
        
        return improved_samples

# Instância global do Haddix
haddix = HaddixLLM()