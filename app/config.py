from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(os.getcwd(), ".env"), extra="ignore")

    bearer_token: str = "dev-token"
    agnes_api_key: str = ""
    bilibili_sessdata: str = ""
    data_dir: str = "./data"
    # LLM 配置（OpenAI 兼容接口，默认 Agnes AI）
    llm_base_url: str = "https://api.agnes-ai.cn/v1/chat/completions"
    llm_model: str = "agnes-2.5-flash"


settings = Settings()
