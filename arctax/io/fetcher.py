"""
Fetcher para obter dados de fontes externas (GitHub, local)
"""

import httpx
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse, urljoin


class SourceFetcher:
    """Fetcher para obter taxonomia de diferentes fontes"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".arctax" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, source: Union[str, Path]) -> Path:
        """
        Fetcha fonte e retorna caminho local
        
        Args:
            source: URL do GitHub ou caminho local
            
        Returns:
            Path para o diretório local com os dados
        """
        
        if isinstance(source, Path) or not self._is_url(str(source)):
            # Fonte local
            return Path(source)
        
        # Fonte remota (GitHub)
        return self._fetch_github_repo(str(source))
    
    def _is_url(self, source: str) -> bool:
        """Verifica se fonte é uma URL"""
        try:
            result = urlparse(source)
            return bool(result.scheme and result.netloc)
        except:
            return False
    
    def _fetch_github_repo(self, url: str) -> Path:
        """Baixa repositório do GitHub"""
        
        # Parse da URL do GitHub
        if "github.com" not in url:
            raise ValueError(f"URL não é do GitHub: {url}")
        
        # Extrai owner/repo da URL
        parts = urlparse(url).path.strip('/').split('/')
        if len(parts) < 2:
            raise ValueError(f"URL do GitHub inválida: {url}")
        
        owner, repo = parts[0], parts[1]
        
        # Monta URL do zip
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
        
        # Nome do cache
        cache_name = f"{owner}_{repo}_main"
        cache_path = self.cache_dir / cache_name
        
        # Se já existe cache, retorna
        if cache_path.exists():
            print(f"Usando cache: {cache_path}")
            return cache_path
        
        # Baixa e extrai
        return self._download_and_extract(zip_url, cache_path, f"{repo}-main")
    
    def _download_and_extract(self, zip_url: str, cache_path: Path, expected_folder: str) -> Path:
        """Baixa ZIP e extrai para cache"""
        
        print(f"Baixando {zip_url}...")
        
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(zip_url)
            response.raise_for_status()
        
        # Extrai ZIP
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = Path(tmp_file.name)
        
        try:
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Extrai para pasta temporária
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(temp_dir)
                    
                    # Move pasta extraída para cache
                    extracted_path = Path(temp_dir) / expected_folder
                    if not extracted_path.exists():
                        # Procura primeira pasta
                        folders = [p for p in Path(temp_dir).iterdir() if p.is_dir()]
                        if folders:
                            extracted_path = folders[0]
                        else:
                            raise FileNotFoundError("Nenhuma pasta encontrada no ZIP")
                    
                    # Move para cache final
                    import shutil
                    shutil.copytree(extracted_path, cache_path)
        
        finally:
            # Limpa arquivo temporário
            tmp_path.unlink()
        
        print(f"Extraído para: {cache_path}")
        return cache_path
    
    def clear_cache(self) -> None:
        """Limpa cache de downloads"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("Cache limpo!")
    
    def list_cache(self) -> list[Path]:
        """Lista itens no cache"""
        if not self.cache_dir.exists():
            return []
        return [p for p in self.cache_dir.iterdir() if p.is_dir()]