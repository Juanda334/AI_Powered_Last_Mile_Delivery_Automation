import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from AI_Powered_Last_Mile_Delivery_Automation.utils.config_loader import load_config
from langchain_huggingface import HuggingFaceEmbeddings
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.exceptions.exception import (
    DocumentPortalException,
)

logger = get_module_logger("utils.model_loader")


class ApiKeyManager:
    REQUIRED_KEYS = ["OPENAI_API_KEY", "OPENAI_API_BASE"]

    def __init__(self):
        self.api_keys = {}

        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    logger.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            logger.error(f"Missing required API keys: {missing}")
            raise DocumentPortalException("Missing API keys", sys)

        logger.info(f"API keys loaded: {list(self.api_keys.keys())}")

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            logger.info("Running in LOCAL mode: .env loaded")
        else:
            logger.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        logger.info(f"YAML config loaded: {list(self.config.keys())}")

    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            logger.info(f"Loading embedding model: {model_name}")
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
                encode_kwargs={"normalize_embeddings": True},
            )
            return embedding_model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        gen_llm_key = os.getenv("GEN_LLM", "gen_llm")
        eval_llm_key = os.getenv("EVAL_LLM", "eval_llm")

        if gen_llm_key not in llm_block:
            logger.error(f"LLM type not found in config: {gen_llm_key}")
            raise ValueError(f"LLM provider '{gen_llm_key}' not found in config")

        if eval_llm_key not in llm_block:
            logger.error(f"LLM type not found in config: {eval_llm_key}")
            raise ValueError(f"LLM provider '{eval_llm_key}' not found in config")

        gen_llm_config = llm_block[gen_llm_key]
        gen_provider = gen_llm_config.get("provider")
        gen_model_name = gen_llm_config.get("model_name")
        gen_temperature = gen_llm_config.get("temperature")

        logger.info(f"Loading Gen LLM: {gen_provider} - {gen_model_name}")

        eval_llm_config = llm_block[eval_llm_key]
        eval_provider = eval_llm_config.get("provider")
        eval_model_name = eval_llm_config.get("model_name")
        eval_temperature = eval_llm_config.get("temperature")

        logger.info(f"Loading Eval LLM: {eval_provider} - {eval_model_name}")

        gen_llm = ChatOpenAI(
            model=gen_model_name, temperature=gen_temperature
        )  # Generation/resolution/communication agents
        eval_llm = ChatOpenAI(
            model=eval_model_name, temperature=eval_temperature
        )  # Critic agents and coherence evaluator

        return gen_llm, eval_llm

    def load_chromadb_keys(self) -> dict:
        """Load ChromaDB credentials from env vars, falling back to config.yaml."""
        CHROMA_KEYS = ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"]
        keys = {}
        config_chromadb = self.config.get("chromadb", {})

        for key in CHROMA_KEYS:
            val = os.getenv(key) or config_chromadb.get(key)
            if val:
                keys[key] = str(val)

        missing = [k for k in CHROMA_KEYS if k not in keys]
        if missing:
            logger.error(f"Missing ChromaDB keys: {missing}")
            raise DocumentPortalException(f"Missing ChromaDB keys: {missing}", sys)

        logger.info("ChromaDB credentials loaded")
        return keys


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    gen_llm, eval_llm = loader.load_llm()
    print(f"Gen LLM Loaded: {gen_llm}")
    result = gen_llm.invoke("Hello, how are you?")
    print(f"Gen LLM Result: {result.content}")
    result = eval_llm.invoke("Hello, how are you?")
    print(f"Eval LLM Result: {result.content}")
