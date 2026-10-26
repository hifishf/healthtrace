"""单一身份层：user_id 的唯一来源，业务代码不直接读 settings.default_user_id。

单用户先固定默认 UUID；将来接多来源（登录态/设备/多档案）时，只改这里——
把「当前用户是谁」的判定收敛到一处，避免散落在工具/图/端点里硬编码。
"""
from uuid import UUID

from app.config import settings


def get_current_user_id() -> UUID:
    """返回当前用户的 user_id（单用户场景固定为默认用户）。"""
    return settings.default_user_id
