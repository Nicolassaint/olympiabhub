import os
import requests
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class OlympiaAPI:
    """
    A class for interacting with the Olympia API, providing access to LLM and embedding models.

    This class handles both direct API calls and requests through a Nubonyxia proxy, offering
    methods for text generation and embedding creation.

    Args:
        model (str): The model identifier to use for generating responses or embeddings.
        token (str, optional): The API authentication token. If not provided, will attempt to read
            from OLYMPIA_API_KEY or OLYMPIA_API_TOKEN environment variables.
        proxy (str, optional): The proxy URL to use for Nubonyxia requests. If not provided,
            will attempt to read from PROXY environment variable.

    Raises:
        ValueError: If no token is provided and none can be found in environment variables.

    Attributes:
        token (str): The API authentication token.
        model (str): The selected model identifier.
        base_url (str): The base URL for the Olympia API.
        nubonyxia_proxy (str): The configured proxy URL for Nubonyxia requests.
        nubonyxia_user_agent (str): The user agent string used for Nubonyxia requests.

    Example:
        >>> api = OlympiaAPI(model="gpt-3.5", token="your-api-token")
        >>> response = api.Chat("Hello, how are you?")
        >>> embeddings = api.create_embedding(["Text to embed"])
    """

    def __init__(self, model: str, token: str = None, proxy: str = None):
        self.token = (
            token or os.getenv("OLYMPIA_API_KEY") or os.getenv("OLYMPIA_API_TOKEN")
        )
        if not self.token:
            raise ValueError(
                "Token is required. Set OLYMPIA_API_KEY/OLYMPIA_API_TOKEN or pass as parameter."
            )

        self.model = model
        self.base_url = "https://api.olympia.bhub.cloud"
        self.nubonyxia_proxy = proxy or os.getenv("PROXY")
        self.nubonyxia_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self, method: str, endpoint: str, data: Dict = None, use_proxy: bool = False
    ) -> Any:
        url = f"{self.base_url}/{endpoint}"
        session = requests.Session() if use_proxy else requests

        if use_proxy and self.nubonyxia_proxy:
            session.get_adapter("https://").proxy_manager_for(
                f"http://{self.nubonyxia_proxy}"
            ).proxy_headers["User-Agent"] = self.nubonyxia_user_agent
            session.proxies.update(
                {"http": self.nubonyxia_proxy, "https": self.nubonyxia_proxy}
            )

        try:
            response = session.request(
                method=method, url=url, headers=self._get_headers(), json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise ConnectionError(f"Failed to connect: {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error: {e}")
            raise TimeoutError(f"Request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error: {e}, Response: {response.text if response else 'No response'}"
            )
            raise ValueError(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise RuntimeError(f"Request failed: {e}")

    def Chat(self, prompt: str) -> Dict[str, Any]:
        return self._make_request(
            method="POST",
            endpoint="generate",
            data={"model": self.model, "prompt": prompt},
        )

    def ChatNubonyxia(self, prompt: str) -> Dict[str, Any]:
        return self._make_request(
            method="POST",
            endpoint="generate",
            data={"model": self.model, "prompt": prompt},
            use_proxy=True,
        )

    def create_embedding(self, texts: List[str]) -> Dict[str, Any]:
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Texts must be a non-empty list of strings")
        return self._make_request(
            method="POST",
            endpoint="embedding",
            data={"model": self.model, "texts": texts},
        )

    def create_embedding_nubonyxia(self, texts: List[str]) -> Dict[str, Any]:
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Texts must be a non-empty list of strings")
        return self._make_request(
            method="POST",
            endpoint="embedding",
            data={"model": self.model, "texts": texts},
            use_proxy=True,
        )

    def get_llm_models(self, use_proxy: bool = False) -> List[str]:
        """Get available LLM models. Set use_proxy=True to use Nubonyxia proxy."""
        return self._make_request(
            method="GET", 
            endpoint="modeles",
            use_proxy=use_proxy
        )["modèles"]

    def get_embedding_models(self, use_proxy: bool = False) -> List[str]:
        """Get available embedding models. Set use_proxy=True to use Nubonyxia proxy."""
        return self._make_request(
            method="GET", 
            endpoint="embedding/models",
            use_proxy=use_proxy
        )["modèles"]
