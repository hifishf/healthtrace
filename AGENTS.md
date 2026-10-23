# HealthTrace 项目约定

## 数据库建表规范

- 所有 `CREATE TABLE` 建表语句必须带注释：
  - 每个表用 `COMMENT ON TABLE ...` 说明用途；
  - 每个字段用 `COMMENT ON COLUMN ...` 说明含义（含枚举取值、计算口径等）。
- 修改已有表结构时，同步新增 / 更新对应字段注释。
- 建表脚本统一维护在 Alembic 迁移（`alembic/versions/`），并保证可幂等重放
  （`upgrade head` / `downgrade -1`）；`schema/init.sql` 自第 6 周起仅作历史基线参考，不再手改。

示例见 `schema/init.sql`。
