"""
WebAPI Agent キャッシュ管理システム

PydanticAI Agent の効率的なキャッシュとライフサイクル管理を提供する。
既存のModelLoadシステムとの整合性を保ちながら、WebAPI特有の要件に対応。
"""

import time
from typing import Any, ClassVar

from pydantic_ai import Agent

from .utils import logger


class WebApiAgentCache:
    """WebAPI Agent専用のキャッシュ管理クラス

    ModelLoadクラスのパターンを参考に、PydanticAI Agentの効率的な
    キャッシュ管理を提供する。軽量なAgentの特性を活かした設計。
    """

    # --- クラス変数でキャッシュ状態を管理 ---
    _AGENT_CACHE: ClassVar[dict[str, Agent]] = {}
    _AGENT_LAST_USED: ClassVar[dict[str, float]] = {}
    _AGENT_CONFIG_HASH: ClassVar[dict[str, str]] = {}
    _MAX_CACHE_SIZE: ClassVar[int] = 50  # Agentは軽量なので多めにキャッシュ

    @classmethod
    def get_agent(cls, cache_key: str, agent_creator: callable, config_hash: str | None = None) -> Agent:
        """Agentを取得またはキャッシュから返す

        Args:
            cache_key: キャッシュキー（通常は model_name + provider_name の組み合わせ）
            agent_creator: Agent作成関数（キャッシュミス時に呼ばれる）
            config_hash: 設定のハッシュ値（設定変更検出用）

        Returns:
            PydanticAI Agent インスタンス
        """
        # 設定変更チェック
        if cls._is_config_changed(cache_key, config_hash):
            cls._invalidate_agent(cache_key)

        # キャッシュヒット
        if cache_key in cls._AGENT_CACHE:
            cls._update_last_used(cache_key)
            logger.debug(f"WebAPI Agent キャッシュヒット: {cache_key}")
            return cls._AGENT_CACHE[cache_key]

        # キャッシュミス - 新規作成
        logger.debug(f"WebAPI Agent キャッシュミス: {cache_key}")

        # キャッシュサイズチェック
        if len(cls._AGENT_CACHE) >= cls._MAX_CACHE_SIZE:
            cls._evict_least_recently_used()

        # 新規Agent作成
        try:
            agent = agent_creator()
            cls._store_agent(cache_key, agent, config_hash)
            logger.info(f"WebAPI Agent 新規作成してキャッシュ: {cache_key}")
            return agent

        except Exception as e:
            logger.error(f"WebAPI Agent 作成失敗: {cache_key}, エラー: {e}")
            raise

    @classmethod
    def _is_config_changed(cls, cache_key: str, config_hash: str | None) -> bool:
        """設定変更をチェックする"""
        if config_hash is None:
            return False

        cached_hash = cls._AGENT_CONFIG_HASH.get(cache_key)
        return cached_hash is not None and cached_hash != config_hash

    @classmethod
    def _invalidate_agent(cls, cache_key: str) -> None:
        """指定されたAgentをキャッシュから削除する"""
        if cache_key in cls._AGENT_CACHE:
            del cls._AGENT_CACHE[cache_key]
            logger.debug(f"WebAPI Agent キャッシュ無効化: {cache_key}")

        if cache_key in cls._AGENT_LAST_USED:
            del cls._AGENT_LAST_USED[cache_key]

        if cache_key in cls._AGENT_CONFIG_HASH:
            del cls._AGENT_CONFIG_HASH[cache_key]

    @classmethod
    def _update_last_used(cls, cache_key: str) -> None:
        """最終使用時刻を更新する"""
        cls._AGENT_LAST_USED[cache_key] = time.time()

    @classmethod
    def _store_agent(cls, cache_key: str, agent: Agent, config_hash: str | None) -> None:
        """Agentをキャッシュに保存する"""
        cls._AGENT_CACHE[cache_key] = agent
        cls._update_last_used(cache_key)

        if config_hash:
            cls._AGENT_CONFIG_HASH[cache_key] = config_hash

    @classmethod
    def _evict_least_recently_used(cls) -> None:
        """LRU戦略で最も古いAgentを削除する"""
        if not cls._AGENT_LAST_USED:
            return

        # 最も古い使用時刻のキーを見つける
        oldest_key = min(cls._AGENT_LAST_USED.items(), key=lambda x: x[1])[0]

        logger.debug(f"WebAPI Agent LRU削除: {oldest_key}")
        cls._invalidate_agent(oldest_key)

    @classmethod
    def clear_cache(cls) -> None:
        """キャッシュをすべてクリアする"""
        cleared_count = len(cls._AGENT_CACHE)
        cls._AGENT_CACHE.clear()
        cls._AGENT_LAST_USED.clear()
        cls._AGENT_CONFIG_HASH.clear()

        if cleared_count > 0:
            logger.info(f"WebAPI Agent キャッシュクリア: {cleared_count}個のAgentを削除")

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """キャッシュの状態情報を取得する"""
        return {
            "cached_agents": list(cls._AGENT_CACHE.keys()),
            "cache_size": len(cls._AGENT_CACHE),
            "max_cache_size": cls._MAX_CACHE_SIZE,
            "last_used_times": dict(cls._AGENT_LAST_USED),
        }

    @classmethod
    def set_max_cache_size(cls, size: int) -> None:
        """最大キャッシュサイズを設定する"""
        if size < 1:
            raise ValueError("最大キャッシュサイズは1以上である必要があります")

        old_size = cls._MAX_CACHE_SIZE
        cls._MAX_CACHE_SIZE = size

        # 新しいサイズを超えている場合は古いものから削除
        while len(cls._AGENT_CACHE) > cls._MAX_CACHE_SIZE:
            cls._evict_least_recently_used()

        logger.info(f"WebAPI Agent 最大キャッシュサイズ変更: {old_size} → {size}")


def create_cache_key(model_name: str, provider_name: str, api_model_id: str | None = None) -> str:
    """Agent用のキャッシュキーを生成する

    Args:
        model_name: 設定ファイルでのモデル名
        provider_name: プロバイダー名 (openai, google, anthropic など)
        api_model_id: 実際のAPIモデルID（オプション）

    Returns:
        キャッシュキー文字列
    """
    if api_model_id:
        return f"{provider_name}:{model_name}:{api_model_id}"
    else:
        return f"{provider_name}:{model_name}"


def create_config_hash(config_dict: dict[str, Any]) -> str:
    """設定辞書からハッシュ値を生成する

    Args:
        config_dict: Agent設定の辞書

    Returns:
        設定のハッシュ値
    """
    import hashlib
    import json

    # 辞書をソートしてJSON文字列化してハッシュ
    config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]  # 8文字で十分
