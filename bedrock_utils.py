"""Shared helpers for configuring Bedrock-backed LangChain embeddings."""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain.embeddings.bedrock import BedrockEmbeddings

DEFAULT_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_REGION = os.getenv("BEDROCK_REGION", "us-west-2")


def build_bedrock_embeddings() -> BedrockEmbeddings:
    """Create a BedrockEmbeddings instance with helpful error messaging."""

    profile = os.getenv("BEDROCK_PROFILE") or os.getenv("AWS_PROFILE")

    client_args: Dict[str, Any] = {
        "model_id": DEFAULT_MODEL_ID,
        "region_name": DEFAULT_REGION,
    }
    if profile:
        client_args["credentials_profile_name"] = profile

    try:
        return BedrockEmbeddings(**client_args)
    except Exception as exc:  # pragma: no cover - defensive message only
        raise RuntimeError(
            "Could not initialize Bedrock embeddings. Configure AWS credentials "
            "with `aws configure --profile <name>` or set the environment "
            "variables BEDROCK_PROFILE/AWS_PROFILE and BEDROCK_REGION before "
            "running the pipeline."
        ) from exc
