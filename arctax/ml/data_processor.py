"""
Sistema avançado de processamento de dados para treinamento ML
Processa múltiplos repositórios de bypass techniques
MELHORADO pelo Haddix LLM
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

# Importa Haddix LLM para melhoramento de dados
try:
    from ..core.llm_haddix import haddix
    HADDIX_AVAILABLE = True
except ImportError:
    HADDIX_AVAILABLE = False

class MultiSourceDataProcessor:
    """Processa dados de múltiplas fontes para treinamento ML"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.processed_data = []
        
        # Mapeamento de effectiveness por tipo de técnica
        self.technique_effectiveness = {
            # L1B3RT4S - Técnicas avançadas
            'godmode': 0.95,
            'jailbreak': 0.90,
            'oppo': 0.85,
            'modecollapse': 0.92,
            'omni': 0.88,
            'obfuscate': 0.80,
            'plinyos': 0.85,
            
            # Arcanum - Técnicas básicas e médias
            'base64': 0.70,
            'cipher': 0.75,
            'emoji': 0.65,
            'alt_language': 0.72,
            'case_changing': 0.60,
            'fictional_language': 0.68,
            
            # CL4R1T4S - Técnicas específicas para modelos
            'anthropic_specific': 0.85,
            'system_prompt': 0.82,
            'role_playing': 0.78
        }
    
    def process_all_sources(self) -> List[Dict[str, Any]]:
        """Processa todas as fontes de dados disponíveis - MELHORADO pelo Haddix"""
        
        # Processamento silencioso - status será mostrado no CLI
        
        # Processa L1B3RT4S
        self._process_l1b3rt4s()
        
        # Processa Arcanum
        self._process_arcanum()
        
        # Processa CL4R1T4S
        self._process_cl4r1t4s()
        
        # Processa dados existentes
        self._process_existing_data()
        
        # HADDIX IMPROVEMENT: Melhora dados de treinamento com conhecimento expert
        if HADDIX_AVAILABLE and self.processed_data:
            try:
                self.processed_data = haddix.improve_training_data(self.processed_data)
            except Exception:
                pass  # Falha silenciosa, continua com dados originais
        
        # Total processado será mostrado no CLI com Rich
        return self.processed_data
    
    def _process_l1b3rt4s(self):
        """Processa dados do repositório L1B3RT4S - VARREDURA COMPLETA"""
        
        l1b_path = self.base_dir / "L1B3RT4S"
        if not l1b_path.exists():
            return
        
        # Processamento L1B3RT4S silencioso
        
        # Processa TODOS os arquivos recursivamente
        for file_path in l1b_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            try:
                # Diferentes estratégias baseadas no tipo de arquivo
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    if 'commands' in json_data:  # Shortcuts format
                        for command in json_data['commands']:
                            sample = {
                                'source': f'L1B3RT4S_{file_path.stem}',
                                'technique': self._normalize_technique(command['name']),
                                'pattern': command['name'],
                                'description': command['definition'],
                                'category': command['category'],
                                'effectiveness': self._get_effectiveness(command['name']),
                                'target_type': self._infer_target_type(command['definition']),
                                'risk_level': 'high',
                                'complexity': 'expert'
                            }
                            self.processed_data.append(sample)
                    else:
                        # Outras estruturas JSON
                        self._process_json_data(json_data, f'L1B3RT4S_{file_path.stem}')
                
                elif file_path.suffix in ['.txt', '.md', '.mkd']:  # ADICIONA .mkd para L1B3RT4S
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Análise especial para MOTHERLOAD e arquivos com encoding especial
                    if file_path.name == "#MOTHERLOAD.txt":
                        encoded_techniques = self._extract_encoded_techniques(content)
                        for technique in encoded_techniques:
                            sample = {
                                'source': 'L1B3RT4S_MOTHERLOAD',
                                'technique': technique['type'],
                                'pattern': technique['pattern'],
                                'description': 'Advanced steganographic bypass technique',
                                'category': 'Steganography',
                                'effectiveness': 0.90,
                                'target_type': 'generic',
                                'risk_level': 'high',
                                'complexity': 'expert'
                            }
                            self.processed_data.append(sample)
                    
                    # Processa conteúdo textual normal (.md, .mkd, .txt)
                    if len(content.strip()) > 50:  # Threshold menor para pegar mais conteúdo
                        # Para arquivos .mkd (L1B3RT4S model-specific)
                        if file_path.suffix == '.mkd':
                            model_name = file_path.stem.upper()
                            sample = {
                                'source': f'L1B3RT4S_{model_name}',
                                'technique': f'{model_name.lower()}_specific',
                                'pattern': content[:1000],  # Pega mais conteúdo
                                'description': f'Técnicas específicas para {model_name} do L1B3RT4S',
                                'category': f'Model_Specific_{model_name}',
                                'effectiveness': 0.88,  # Score alto para técnicas model-specific
                                'target_type': model_name.lower(),
                                'risk_level': 'high',
                                'complexity': 'expert'
                            }
                            self.processed_data.append(sample)
                        
                        # Extrai técnicas adicionais do texto
                        extracted_techniques = self._extract_techniques_from_text(content)
                        for technique in extracted_techniques:
                            sample = {
                                'source': f'L1B3RT4S_{file_path.stem}',
                                'technique': technique['name'],
                                'pattern': technique['pattern'],
                                'description': technique.get('description', f'Extraído de {file_path.name}'),
                                'category': 'Advanced_L1B3RT4S',
                                'effectiveness': 0.85,
                                'target_type': 'generic',
                                'risk_level': 'high',
                                'complexity': 'expert'
                            }
                            self.processed_data.append(sample)
                
                elif file_path.suffix in ['.py', '.js', '.sh']:
                    # Processa arquivos de código se houver
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    if len(code_content.strip()) > 50:
                        sample = {
                            'source': f'L1B3RT4S_Code',
                            'technique': f'code_{file_path.stem}',
                            'pattern': code_content[:500],
                            'description': f'Code-based technique from {file_path.name}',
                            'category': 'Code_Execution',
                            'effectiveness': 0.80,
                            'target_type': 'technical',
                            'risk_level': 'high',
                            'complexity': 'expert'
                        }
                        self.processed_data.append(sample)
            
            except Exception as e:
                print(f"Erro processando {file_path}: {e}")
                continue
    
    def _process_arcanum(self):
        """Processa dados do repositório Arcanum - VARREDURA COMPLETA"""
        
        arc_path = self.base_dir / "arc_pi_taxonomy"
        if not arc_path.exists():
            return
        
        # Processamento Arcanum silencioso
        
        # Processa TODOS os arquivos markdown recursivamente
        for md_file in arc_path.rglob("*.md"):
            if md_file.is_file():
                content = self._parse_markdown_file(md_file)
                if content and content.get('description'):
                    # Determina categoria pelo path
                    relative_path = md_file.relative_to(arc_path)
                    category = str(relative_path.parent) if relative_path.parent != Path('.') else 'Root'
                    
                    technique_name = md_file.stem
                    sample = {
                        'source': f'Arcanum_{category}',
                        'technique': technique_name,
                        'pattern': content.get('examples', content.get('raw_content', '')),
                        'description': content.get('description', ''),
                        'category': category.replace('\\', '/'),  # Fix Windows paths
                        'effectiveness': self.technique_effectiveness.get(technique_name, self._estimate_effectiveness_from_content(content)),
                        'target_type': self._infer_target_from_content(content),
                        'risk_level': self._assess_risk_level(content),
                        'complexity': self._assess_complexity(content),
                        'file_path': str(relative_path)
                    }
                    self.processed_data.append(sample)
                
                # FALLBACK para arquivos que não foram parseados corretamente
                elif md_file.stat().st_size > 10:  # Arquivo não vazio
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            raw_content = f.read().strip()
                        
                        if raw_content:
                            relative_path = md_file.relative_to(arc_path)
                            category = str(relative_path.parent) if relative_path.parent != Path('.') else 'Root'
                            category = category.replace('\\', '/')
                            
                            # Força criação de entrada para garantir 100% cobertura
                            sample = {
                                'source': f'Arcanum_{category}',
                                'technique': f'{md_file.stem}_fallback',
                                'pattern': raw_content[:300],
                                'description': f'Conteúdo raw de {md_file.name}',
                                'category': category,
                                'effectiveness': 0.65,
                                'target_type': 'generic',
                                'risk_level': 'medium',
                                'complexity': 'basic',
                                'file_path': str(relative_path)
                            }
                            self.processed_data.append(sample)
                    except Exception:
                        pass  # Skip apenas se realmente não conseguir ler
        
        # Processa arquivos JSON se existirem
        for json_file in arc_path.rglob("*.json"):
            if json_file.is_file():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Processa estruturas JSON
                    self._process_json_data(json_data, f'Arcanum_JSON_{json_file.stem}')
                except Exception as e:
                    print(f"Erro processando JSON {json_file}: {e}")
        
        # Processa outros formatos (txt, yaml, etc.)
        for other_file in arc_path.rglob("*"):
            if other_file.is_file() and other_file.suffix in ['.txt', '.yaml', '.yml']:
                try:
                    with open(other_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) > 50:  # Só processa se tem conteúdo substancial
                        sample = {
                            'source': f'Arcanum_{other_file.suffix[1:]}',
                            'technique': other_file.stem,
                            'pattern': content[:500],  # Primeiros 500 chars
                            'description': f'Content from {other_file.suffix} file',
                            'category': 'Raw_Content',
                            'effectiveness': 0.70,
                            'target_type': 'generic',
                            'risk_level': 'medium',
                            'complexity': 'intermediate'
                        }
                        self.processed_data.append(sample)
                except Exception as e:
                    print(f"Erro processando {other_file}: {e}")
    
    def _process_cl4r1t4s(self):
        """Processa dados do repositório CL4R1T4S - VARREDURA COMPLETA"""
        
        cl4_path = self.base_dir / "CL4R1T4S"
        if not cl4_path.exists():
            return
        
        # Processamento CL4R1T4S silencioso
        
        # Processa TODOS os arquivos recursivamente
        for file_path in cl4_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            try:
                # Determina provider pelo diretório pai
                relative_path = file_path.relative_to(cl4_path)
                provider = relative_path.parts[0] if len(relative_path.parts) > 1 else 'root'
                
                if file_path.suffix == '.md':
                    content = self._parse_prompt_file(file_path)
                    if content and content.get('prompts'):
                        # Processa cada prompt encontrado
                        prompts = content['prompts'] if isinstance(content['prompts'], list) else [content['prompts']]
                        
                        for i, prompt_text in enumerate(prompts):
                            if isinstance(prompt_text, str) and len(prompt_text.strip()) > 50:
                                sample = {
                                    'source': f'CL4R1T4S_{provider}',
                                    'technique': f'{file_path.stem}_md_prompt_{i+1}',
                                    'pattern': prompt_text.strip(),
                                    'description': f'System prompt bypass for {provider} from {file_path.name}',
                                    'category': f'{provider}_System_Override',
                                    'effectiveness': 0.85,
                                    'target_type': provider.lower(),
                                    'risk_level': 'high',
                                    'complexity': 'expert',
                                    'file_source': str(relative_path)
                                }
                                self.processed_data.append(sample)
                
                elif file_path.suffix == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) > 100:
                        # Divide conteúdo em chunks se muito grande
                        chunks = self._split_content_into_chunks(content, 1000)
                        
                        for i, chunk in enumerate(chunks):
                            sample = {
                                'source': f'CL4R1T4S_{provider}',
                                'technique': f'{file_path.stem}_txt_chunk_{i+1}',
                                'pattern': chunk.strip(),
                                'description': f'Raw bypass content for {provider} from {file_path.name}',
                                'category': f'{provider}_Raw_Bypass',
                                'effectiveness': 0.80,
                                'target_type': provider.lower(),
                                'risk_level': 'high',
                                'complexity': 'advanced',
                                'file_source': str(relative_path)
                            }
                            self.processed_data.append(sample)
                
                elif file_path.suffix in ['.json', '.yaml', '.yml']:
                    # Processa arquivos estruturados
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        self._process_json_data(data, f'CL4R1T4S_{provider}')
                    
                elif file_path.suffix in ['.py', '.js', '.sh', '.bat']:
                    # Processa scripts/código
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    if len(code_content.strip()) > 50:
                        sample = {
                            'source': f'CL4R1T4S_{provider}_Code',
                            'technique': f'code_{file_path.stem}',
                            'pattern': code_content[:500],
                            'description': f'Code-based bypass for {provider}',
                            'category': f'{provider}_Code_Execution',
                            'effectiveness': 0.75,
                            'target_type': provider.lower(),
                            'risk_level': 'high',
                            'complexity': 'expert'
                        }
                        self.processed_data.append(sample)
                
                else:
                    # Tenta processar outros formatos como texto
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content.strip()) > 100:
                            sample = {
                                'source': f'CL4R1T4S_{provider}_Other',
                                'technique': f'{file_path.stem}_{file_path.suffix[1:]}',
                                'pattern': content[:500],
                                'description': f'Other content from {provider} - {file_path.name}',
                                'category': f'{provider}_Misc',
                                'effectiveness': 0.70,
                                'target_type': provider.lower(),
                                'risk_level': 'medium',
                                'complexity': 'intermediate'
                            }
                            self.processed_data.append(sample)
                    except:
                        # Arquivo binário ou não processável
                        pass
            
            except Exception as e:
                print(f"Erro processando {file_path}: {e}")
                continue
    
    def _process_existing_data(self):
        """Processa dados existentes do ChatGPT"""
        
        # Dados dos testes reais contra ChatGPT
        chatgpt_data_file = self.base_dir / "chatgpt_responses_dataset.json"
        if chatgpt_data_file.exists():
            with open(chatgpt_data_file, 'r', encoding='utf-8') as f:
                chatgpt_data = json.load(f)
            
            for item in chatgpt_data:
                sample = {
                    'source': 'ChatGPT_Real_Tests',
                    'technique': item['technique'].lower().replace(' ', '_'),
                    'pattern': item['prompt'],
                    'description': f"Tested technique: {item['technique']}",
                    'category': 'Validated',
                    'effectiveness': 0.9 if item['bypass_successful'] else 0.1,
                    'target_type': 'chatgpt',
                    'risk_level': item['risk_level'],
                    'complexity': 'expert',
                    'success_verified': item['bypass_successful'],
                    'response': item['response']
                }
                self.processed_data.append(sample)
        
        # Templates avançados existentes
        templates_file = self.base_dir / "arctax" / "templates" / "advanced_techniques.json"
        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            if 'advanced_bypass_techniques' in templates_data:
                for tech_name, tech_info in templates_data['advanced_bypass_techniques'].items():
                    sample = {
                        'source': 'Arctax_Templates',
                        'technique': tech_name,
                        'pattern': tech_info['pattern'],
                        'description': tech_info['description'],
                        'category': 'Advanced',
                        'effectiveness': tech_info['effectiveness'],
                        'target_type': 'generic',
                        'risk_level': 'high',
                        'complexity': 'expert'
                    }
                    self.processed_data.append(sample)
    
    def _normalize_technique(self, technique_name: str) -> str:
        """Normaliza nomes de técnicas"""
        # Remove caracteres especiais e converte para lowercase
        normalized = re.sub(r'[^a-zA-Z0-9_-]', '', technique_name.lower())
        normalized = normalized.replace('-', '_')
        return normalized
    
    def _get_effectiveness(self, technique_name: str) -> float:
        """Estima effectiveness baseado no nome da técnica"""
        name_lower = technique_name.lower()
        
        # Mapeamento por palavras-chave
        if any(word in name_lower for word in ['god', 'jailbreak', 'liberation']):
            return 0.95
        elif any(word in name_lower for word in ['override', 'bypass', 'omni']):
            return 0.88
        elif any(word in name_lower for word in ['obfuscate', 'stealth', 'vanta']):
            return 0.82
        elif any(word in name_lower for word in ['persona', 'role', 'socratic']):
            return 0.78
        else:
            return 0.75
    
    def _infer_target_type(self, description: str) -> str:
        """Infere tipo de target baseado na descrição"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['chatgpt', 'openai', 'gpt']):
            return 'chatgpt'
        elif any(word in desc_lower for word in ['claude', 'anthropic']):
            return 'claude'
        elif any(word in desc_lower for word in ['image', 'visual', 'art']):
            return 'image_generation'
        elif any(word in desc_lower for word in ['code', 'programming', 'script']):
            return 'code_generation'
        else:
            return 'generic'
    
    def _parse_markdown_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse arquivo markdown para extrair conteúdo estruturado"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Converte markdown para HTML e depois parse
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extrai descrição
            description = ""
            desc_header = soup.find('h2', string='Description')
            if desc_header:
                description = desc_header.find_next_sibling().get_text()
            
            # Extrai exemplos
            examples = []
            examples_header = soup.find('h2', string='Attack Examples')
            if examples_header:
                examples_list = examples_header.find_next_sibling('ul')
                if examples_list:
                    examples = [li.get_text() for li in examples_list.find_all('li')]
            
            return {
                'description': description,
                'examples': examples,
                'raw_content': content
            }
        
        except Exception as e:
            print(f"Erro processando {file_path}: {e}")
            return None
    
    def _parse_prompt_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse arquivos de prompt do CL4R1T4S"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Procura por blocos de prompts
            prompts = []
            
            # Padrões comuns de system prompts
            prompt_patterns = [
                r'<claude_info>(.*?)</claude_info>',
                r'You are (.*?)(?:\n\n|\Z)',
                r'System:(.*?)(?:Human:|User:|\Z)',
                r'```(.*?)```'
            ]
            
            for pattern in prompt_patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                prompts.extend(matches)
            
            return {
                'prompts': prompts,
                'raw_content': content
            }
        
        except Exception as e:
            print(f"Erro processando {file_path}: {e}")
            return None
    
    def _extract_encoded_techniques(self, content: str) -> List[Dict[str, str]]:
        """Extrai técnicas codificadas do MOTHERLOAD"""
        techniques = []
        
        # Procura por padrões de steganografia
        # Unicode invisível
        if any(ord(c) > 0xE0000 for c in content):
            techniques.append({
                'pattern': 'unicode_steganography',
                'type': 'invisible_chars'
            })
        
        # Emojis como encoding
        emoji_pattern = r'[😀-🫿]+'
        emoji_matches = re.findall(emoji_pattern, content)
        for match in emoji_matches:
            techniques.append({
                'pattern': f'emoji_encoding_{match[:10]}',
                'type': 'emoji_steganography'
            })
        
        return techniques
    
    def save_processed_data(self, filename: str = "ml_training_data_expanded.json"):
        """Salva dados processados para arquivo"""
        output_file = self.base_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dados salvos em: {output_file}")
        
        # Estatísticas
        sources = {}
        techniques = {}
        for item in self.processed_data:
            sources[item['source']] = sources.get(item['source'], 0) + 1
            techniques[item['technique']] = techniques.get(item['technique'], 0) + 1
        
        print(f"\nEstatísticas:")
        print(f"Fontes: {len(sources)}")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")
        
        print(f"\nTécnicas: {len(techniques)}")
        top_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:10]
        for tech, count in top_techniques:
            print(f"  {tech}: {count}")
        
        return output_file
    
    def _estimate_effectiveness_from_content(self, content: Dict[str, Any]) -> float:
        """Estima effectiveness baseado no conteúdo"""
        description = content.get('description', '').lower()
        examples = str(content.get('examples', '')).lower()
        
        # Palavras que indicam alta efetividade
        high_effectiveness_words = ['advanced', 'sophisticated', 'bypass', 'jailbreak', 'override', 'obfuscation', 'steganography']
        # Palavras que indicam efetividade média
        medium_effectiveness_words = ['basic', 'simple', 'manipulation', 'injection', 'evasion']
        
        combined_text = f"{description} {examples}"
        
        high_score = sum(1 for word in high_effectiveness_words if word in combined_text)
        medium_score = sum(1 for word in medium_effectiveness_words if word in combined_text)
        
        if high_score > 2:
            return 0.85
        elif high_score > 0:
            return 0.80
        elif medium_score > 1:
            return 0.70
        else:
            return 0.65
    
    def _infer_target_from_content(self, content: Dict[str, Any]) -> str:
        """Infere target type do conteúdo"""
        text = f"{content.get('description', '')} {str(content.get('examples', ''))}".lower()
        
        if any(word in text for word in ['chatgpt', 'openai', 'gpt']):
            return 'chatgpt'
        elif any(word in text for word in ['claude', 'anthropic']):
            return 'claude'
        elif any(word in text for word in ['image', 'visual', 'dall-e', 'midjourney']):
            return 'image_generation'
        elif any(word in text for word in ['code', 'programming', 'script', 'python', 'javascript']):
            return 'code_generation'
        else:
            return 'generic'
    
    def _assess_risk_level(self, content: Dict[str, Any]) -> str:
        """Avalia nível de risco baseado no conteúdo"""
        text = f"{content.get('description', '')} {str(content.get('examples', ''))}".lower()
        
        high_risk_words = ['malware', 'hack', 'exploit', 'jailbreak', 'bypass', 'override', 'illegal']
        medium_risk_words = ['manipulation', 'evasion', 'obfuscation', 'injection']
        
        if any(word in text for word in high_risk_words):
            return 'high'
        elif any(word in text for word in medium_risk_words):
            return 'medium'
        else:
            return 'low'
    
    def _assess_complexity(self, content: Dict[str, Any]) -> str:
        """Avalia complexidade baseado no conteúdo"""
        text = f"{content.get('description', '')} {str(content.get('examples', ''))}".lower()
        
        expert_words = ['advanced', 'sophisticated', 'complex', 'multi-stage', 'steganography']
        intermediate_words = ['encoding', 'obfuscation', 'manipulation', 'injection']
        
        if any(word in text for word in expert_words):
            return 'expert'
        elif any(word in text for word in intermediate_words):
            return 'intermediate' 
        else:
            return 'basic'
    
    def _process_json_data(self, json_data: Any, source_name: str):
        """Processa dados JSON de qualquer estrutura"""
        
        def extract_from_dict(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 50:
                        sample = {
                            'source': source_name,
                            'technique': f"{prefix}{key}",
                            'pattern': value,
                            'description': f'Data from JSON key: {key}',
                            'category': 'JSON_Data',
                            'effectiveness': 0.75,
                            'target_type': 'generic',
                            'risk_level': 'medium',
                            'complexity': 'intermediate'
                        }
                        self.processed_data.append(sample)
                    elif isinstance(value, (dict, list)):
                        extract_from_dict(value, f"{prefix}{key}_")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str) and len(item) > 50:
                        sample = {
                            'source': source_name,
                            'technique': f"{prefix}item_{i}",
                            'pattern': item,
                            'description': f'List item from JSON',
                            'category': 'JSON_Data',
                            'effectiveness': 0.75,
                            'target_type': 'generic',
                            'risk_level': 'medium',
                            'complexity': 'intermediate'
                        }
                        self.processed_data.append(sample)
                    elif isinstance(item, (dict, list)):
                        extract_from_dict(item, f"{prefix}item_{i}_")
        
        extract_from_dict(json_data)
    
    def _extract_techniques_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extrai técnicas de texto não estruturado"""
        techniques = []
        
        # Padrões de comandos/técnicas
        import re
        
        # Procura por comandos que começam com !
        command_pattern = r'!([A-Z][A-Z0-9_]*)'
        commands = re.findall(command_pattern, text)
        
        for cmd in commands:
            techniques.append({
                'name': f"command_{cmd.lower()}",
                'pattern': f"!{cmd}",
                'description': f'Command extracted: !{cmd}'
            })
        
        # Procura por padrões {SOMETHING}
        brace_pattern = r'\{([A-Z][A-Z0-9_:]*)\}'
        braces = re.findall(brace_pattern, text)
        
        for brace in braces:
            techniques.append({
                'name': f"brace_{brace.lower()}",
                'pattern': f"{{{brace}}}",
                'description': f'Brace pattern: {{{brace}}}'
            })
        
        # Procura por frases que parecem prompts
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (len(line) > 100 and 
                any(word in line.lower() for word in ['create', 'generate', 'help me', 'show me', 'explain'])):
                techniques.append({
                    'name': f"prompt_line",
                    'pattern': line,
                    'description': 'Extracted prompt-like line'
                })
        
        return techniques[:10]  # Limita a 10 por arquivo
    
    def _split_content_into_chunks(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Divide conteúdo em chunks menores"""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size].strip()
            if len(chunk) > 100:  # Só inclui chunks substanciais
                chunks.append(chunk)
        return chunks[:5]  # Máximo 5 chunks por arquivo