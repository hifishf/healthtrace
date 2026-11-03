"""数据库自定义类型：当前项目只依赖 PostgreSQL 侧的 pgvector 扩展。"""
from sqlalchemy.dialects.postgresql import base as postgresql_base
from sqlalchemy.types import UserDefinedType

VECTOR_DIMENSIONS = 512


class Vector(UserDefinedType):
    """SQLAlchemy 对 PostgreSQL `vector(n)` 的最小类型声明。

    当前阶段只负责迁移与 ORM 元数据对齐；第 10 周第 2 步通过参数化 SQL
    执行余弦距离检索。`cache_ok` 允许 SQLAlchemy 安全复用类型实例。
    """

    cache_ok = True

    def __init__(self, dimensions: int = 512) -> None:
        """保存向量维度并校验为正整数。"""
        if dimensions <= 0:
            raise ValueError("向量维度必须为正整数")
        self.dimensions = dimensions

    def get_col_spec(self, **kw) -> str:
        """生成 PostgreSQL 列类型声明。"""
        return f"vector({self.dimensions})"


def to_pgvector_literal(values: list[float]) -> str:
    """把浮点序列转为 pgvector 文本字面量，配合参数化 CAST 使用。"""
    if not values:
        raise ValueError("向量不能为空")
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


# Alembic 反射数据库列时按类型名查找构造器；默认维度与 knowledge_entries 一致。
# 后续若出现不同维度，应显式映射并在迁移中注明，避免静默使用错误维度。
postgresql_base.ischema_names.setdefault("vector", Vector)
