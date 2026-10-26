"""身份层收敛：业务代码经 get_current_user_id 取用户，单用户默认固定 UUID。"""
from app.config import settings
from app.identity import get_current_user_id


def test_get_current_user_id_returns_default() -> None:
    """单用户默认返回 settings.default_user_id（身份层的唯一出口）。"""
    assert get_current_user_id() == settings.default_user_id
