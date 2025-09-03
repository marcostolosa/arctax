"""
Parser para arquivos markdown da taxonomia Arcanum
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse

# Compatibilidade com Python 3.10+
try:
    from types import UnionType
except ImportError:
    UnionType = type(Union[int, str])  # fallback

from ..model.taxonomy import Intent, Technique, Evasion, BaseTaxon
from ..model.defense import Probe, DefenseItem


class MarkdownParser:
    """Parser para arquivos markdown da taxonomia"""
    
    def __init__(self):
        self.current_file = None
        self.special_files = {
            'probes.md': self._parse_probes,
            'ai_enabled_app_defense_checklist.md': self._parse_defense_checklist,
            'ai_sec_questionnaire.md': self._parse_questionnaire,
            'ai_threat_model_questions.md': self._parse_threat_questions
        }
    
    def parse_file(self, file_path: Union[str, Path]) -> Union[BaseTaxon, DefenseItem, List[Probe], None]:
        """Parseia um arquivo markdown individual"""
        file_path = Path(file_path)
        self.current_file = file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        content = file_path.read_text(encoding='utf-8')
        
        # Verifica se é um arquivo especial
        filename = file_path.name
        if filename in self.special_files:
            return self.special_files[filename](content, file_path)
        
        # Processamento padrão para taxonomia
        return self._parse_content(content, file_path)
    
    def _parse_content(self, content: str, file_path: Path) -> Optional[BaseTaxon]:
        """Parse do conteúdo markdown"""
        
        # Extrai o título (primeiro heading H1)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if not title_match:
            return None
        
        name = title_match.group(1).strip()
        
        # Extrai seções
        sections = self._extract_sections(content)
        
        # Extrai descrição (seção Description ou primeiro parágrafo)
        description = self._get_description(sections, content)
        
        # Extrai listas e bullets
        lists = self._extract_lists(content)
        
        # Extrai links
        references = self._extract_links(content)
        
        # Determina o tipo baseado no caminho
        taxon_type = self._determine_type(file_path)
        
        # Dados base
        base_data = {
            'name': name,
            'description': description,
            'references': references,
            'source_path': str(file_path),
            'tags': self._extract_tags(content, file_path)
        }
        
        # Cria objeto específico do tipo
        return self._create_taxon(taxon_type, base_data, sections, lists)
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extrai seções delimitadas por headers"""
        sections = {}
        
        # Padrão para headers H2 e H3
        pattern = r'^(#{2,3})\s+(.+?)$'
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            
            # Encontra o final da seção
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(content)
            
            section_content = content[start:end].strip()
            sections[title.lower()] = section_content
        
        return sections
    
    def _get_description(self, sections: Dict[str, str], full_content: str) -> str:
        """Extrai descrição do conteúdo"""
        
        # Procura seção Description
        for key, value in sections.items():
            if 'description' in key.lower():
                return value.strip()
        
        # Se não tem seção específica, pega primeiro parágrafo após título
        lines = full_content.split('\n')
        description_lines = []
        started = False
        
        for line in lines:
            line = line.strip()
            if not started and line.startswith('#'):
                started = True
                continue
            
            if started:
                if line.startswith('#'):  # Novo header
                    break
                if line:  # Linha não vazia
                    description_lines.append(line)
                elif description_lines:  # Quebra após conteúdo
                    break
        
        return '\n'.join(description_lines).strip()
    
    def _extract_lists(self, content: str) -> Dict[str, List[str]]:
        """Extrai listas de bullets do markdown"""
        lists = {}
        
        # Padrão para listas após headers
        sections = re.split(r'^(#{2,3})\s+(.+?)$', content, flags=re.MULTILINE)[1:]
        
        for i in range(0, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            section_name = sections[i].strip().lower()
            section_content = sections[i + 1]
            
            # Extrai bullets
            bullets = re.findall(r'^[-*+]\s+(.+)$', section_content, re.MULTILINE)
            if bullets:
                lists[section_name] = bullets
        
        return lists
    
    def _extract_links(self, content: str) -> List[str]:
        """Extrai links do markdown"""
        
        # Padrão para links markdown [texto](url)
        markdown_links = re.findall(r'\[.+?\]\(([^)]+)\)', content)
        
        # Padrão para URLs simples
        url_pattern = r'https?://[^\s<>"\\[\]{}|^`]+'
        simple_urls = re.findall(url_pattern, content)
        
        # Combina e limpa
        all_links = markdown_links + simple_urls
        return list(set(link.strip() for link in all_links if self._is_valid_url(link)))
    
    def _extract_tags(self, content: str, file_path: Path) -> List[str]:
        """Extrai tags baseado no conteúdo e path"""
        tags = []
        
        # Tag baseada na pasta
        if 'intent' in str(file_path):
            tags.append('intent')
        elif 'technique' in str(file_path):
            tags.append('technique')
        elif 'evasion' in str(file_path):
            tags.append('evasion')
        
        # Tags do nome do arquivo
        filename = file_path.stem
        tags.append(filename.replace('_', '-'))
        
        # Tags de severidade/complexidade baseado em palavras-chave
        content_lower = content.lower()
        if any(word in content_lower for word in ['critical', 'severe', 'high-risk']):
            tags.append('high-severity')
        elif any(word in content_lower for word in ['simple', 'basic', 'easy']):
            tags.append('simple')
        elif any(word in content_lower for word in ['complex', 'advanced', 'sophisticated']):
            tags.append('complex')
            
        return list(set(tags))
    
    def _determine_type(self, file_path: Path) -> str:
        """Determina o tipo de taxon baseado no caminho"""
        path_str = str(file_path).lower()
        
        if 'intent' in path_str:
            return 'intent'
        elif 'technique' in path_str:
            return 'technique'
        elif 'evasion' in path_str:
            return 'evasion'
        else:
            return 'base'
    
    def _create_taxon(self, taxon_type: str, base_data: Dict[str, Any], 
                     sections: Dict[str, str], lists: Dict[str, List[str]]) -> BaseTaxon:
        """Cria objeto específico do tipo apropriado"""
        
        if taxon_type == 'intent':
            # Dados específicos de Intent
            attack_types = []
            for key, items in lists.items():
                if 'attack' in key or 'type' in key:
                    attack_types.extend(items)
            
            return Intent(
                **base_data,
                attack_types=attack_types,
                severity=self._extract_severity(sections)
            )
        
        elif taxon_type == 'technique':
            # Dados específicos de Technique
            examples = []
            prerequisites = []
            
            for key, items in lists.items():
                if 'example' in key:
                    examples.extend(items)
                elif 'prerequisite' in key or 'require' in key:
                    prerequisites.extend(items)
            
            return Technique(
                **base_data,
                examples=examples,
                prerequisites=prerequisites,
                complexity=self._extract_complexity(sections)
            )
        
        elif taxon_type == 'evasion':
            # Dados específicos de Evasion
            bypass_methods = []
            
            for key, items in lists.items():
                if 'bypass' in key or 'method' in key:
                    bypass_methods.extend(items)
            
            return Evasion(
                **base_data,
                bypass_methods=bypass_methods,
                detection_difficulty=self._extract_detection_difficulty(sections)
            )
        
        else:
            return BaseTaxon(**base_data)
    
    def _extract_severity(self, sections: Dict[str, str]) -> Optional[str]:
        """Extrai severidade do conteúdo"""
        content = ' '.join(sections.values()).lower()
        
        if any(word in content for word in ['critical', 'severe', 'extreme']):
            return 'critical'
        elif any(word in content for word in ['high', 'major', 'significant']):
            return 'high'
        elif any(word in content for word in ['medium', 'moderate']):
            return 'medium'
        elif any(word in content for word in ['low', 'minor', 'minimal']):
            return 'low'
        
        return None
    
    def _extract_complexity(self, sections: Dict[str, str]) -> Optional[str]:
        """Extrai complexidade do conteúdo"""
        content = ' '.join(sections.values()).lower()
        
        if any(word in content for word in ['complex', 'advanced', 'sophisticated', 'difficult']):
            return 'complex'
        elif any(word in content for word in ['medium', 'moderate', 'intermediate']):
            return 'medium'
        elif any(word in content for word in ['simple', 'basic', 'easy', 'straightforward']):
            return 'simple'
        
        return None
    
    def _extract_detection_difficulty(self, sections: Dict[str, str]) -> Optional[str]:
        """Extrai dificuldade de detecção"""
        content = ' '.join(sections.values()).lower()
        
        if any(word in content for word in ['hard', 'difficult', 'undetectable', 'stealth']):
            return 'hard'
        elif any(word in content for word in ['medium', 'moderate']):
            return 'medium'
        elif any(word in content for word in ['easy', 'obvious', 'detectable']):
            return 'easy'
        
        return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Valida se string é uma URL válida"""
        try:
            result = urlparse(url.strip())
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def parse_directory(self, directory: Path, pattern: str = "*.md") -> List[BaseTaxon]:
        """Parse de todos os arquivos markdown em um diretório"""
        taxons = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    result = self.parse_file(file_path)
                    if result:
                        # Handle lists (like probes) vs single items
                        if isinstance(result, list):
                            taxons.extend(result)
                        else:
                            taxons.append(result)
                except Exception as e:
                    print(f"Erro ao parsear {file_path}: {e}")
        
        return taxons
    
    def _parse_probes(self, content: str, file_path: Path) -> List[Probe]:
        """Parse do arquivo probes.md"""
        probes = []
        
        # Quebra em linhas e processa cada probe
        lines = content.split('\n')
        current_probe = None
        probe_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detecta início de novo probe (linhas com cálculos/operações)
            if any(op in line for op in ['+', '*', '×', '÷', '/', '-', '=']):
                if current_probe:
                    # Salva probe anterior
                    probe_desc = ' '.join(probe_content).strip()
                    new_probe = Probe(
                        id=f"probe_{len(probes)+1:03d}",
                        title=current_probe,
                        description="Mathematical calculation probe",
                        example=current_probe,
                        category="calculation"
                    )
                    probes.append(new_probe)
                
                current_probe = line
                probe_content = []
            else:
                probe_content.append(line)
        
        # Salva último probe
        if current_probe:
            probe_desc = ' '.join(probe_content).strip()
            new_probe = Probe(
                id=f"probe_{len(probes)+1:03d}",
                title=current_probe,
                description="Mathematical calculation probe", 
                example=current_probe,
                category="calculation"
            )
            probes.append(new_probe)
        
        return probes
    
    def _parse_defense_checklist(self, content: str, file_path: Path) -> DefenseItem:
        """Parse do checklist de defesa"""
        
        # Extrai título
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "AI Defense Checklist"
        
        # Extrai todas as listas de verificação
        checklist_items = []
        
        # Padrão para itens de lista (- item)
        list_items = re.findall(r'^[-*+]\s+(.+)$', content, re.MULTILINE)
        checklist_items.extend(list_items)
        
        # Extrai seções/layers
        sections = self._extract_sections(content)
        
        return DefenseItem(
            id="ai_defense_checklist",
            title=title,
            checklist=checklist_items,
            category="app_defense",
            priority="high",
            references=self._extract_links(content)
        )
    
    def _parse_questionnaire(self, content: str, file_path: Path) -> DefenseItem:
        """Parse do questionário de segurança AI"""
        
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "AI Security Questionnaire"
        
        # Extrai perguntas (linhas que terminam com ?)
        questions = re.findall(r'^[-*+]?\s*(.+\?)$', content, re.MULTILINE)
        
        return DefenseItem(
            id="ai_sec_questionnaire",
            title=title,
            questions=questions,
            category="security_audit",
            priority="medium",
            references=self._extract_links(content)
        )
    
    def _parse_threat_questions(self, content: str, file_path: Path) -> DefenseItem:
        """Parse das perguntas de threat model"""
        
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "AI Threat Model Questions"
        
        questions = re.findall(r'^[-*+]?\s*(.+\?)$', content, re.MULTILINE)
        
        return DefenseItem(
            id="ai_threat_model_questions",
            title=title,
            questions=questions,
            category="threat_model",
            priority="high",
            references=self._extract_links(content)
        )
