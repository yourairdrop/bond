# Alpha Suite 公开资料

这是一份面向公开发布的研究仓，不是实盘部署仓。

它只保留对外适合公开的内容：

- 策略方法
- 天气 bond 策略代码
- 天气 bond 回测系统
- 研究框架
- 使用边界
- 对外文章和线程草稿

这份目录已经整理成可以直接上传为独立公开仓的结构。

它不会包含：

- 私钥
- 钱包凭据
- 本地环境密钥文件
- 实盘 state
- 当前更高利润算法的核心逻辑

## 目录

- `alpha_suite/`
  - 公开后的天气 bond 策略代码与研究基础设施
- `alpha_suite/backtest.py`
  - 回测 CLI 入口
- `alpha_suite/backtesting/`
  - 公开版历史回测系统
- `docs/01_PUBLIC_STRATEGY_PLAN.md`
  - 对外公开方案和推荐口径
- `docs/02_STRATEGY_ARTICLE_CN.md`
  - 中文长文，重点介绍策略方法和思路
- `docs/03_HOW_TO_USE_CN.md`
  - 这套框架应该怎么使用
- `docs/04_TWITTER_THREAD_CN.md`
  - 推特 / X 线程草稿
- `docs/05_DATA_BASIS_AND_BOUNDARY.md`
  - 数据边界和公开口径说明

## 这份仓库的定位

这份仓库最准确的定位是：

> 一套真实跑过、真实出现过阶段性盈利窗口、最后又主动停止并切换方向的策略研究框架。

它更适合作为方法论 + 策略公开，而不是作为“复制即赚钱”的成品系统。

## 策略分工

这份仓库目前公开了 3 个策略实现：

- `alpha_suite/strategies/bond_buyer.py`
  - 第一代 `BondBuyer` 主框架
- `alpha_suite/strategies_v2/bond_buyer_v2.py`
  - 后续主要工作框架，也是更成熟的 `BondBuyerV2`
- `alpha_suite/strategies_v2/city_bond.py`
  - 基于 `BondBuyerV2` 的城市校准 dry-run 变体

对外公开的主线内容，核心就是天气 bond 这条研究线。

## 回测

这份公开仓现在包含历史回测子系统，可直接从命令行入口查看：

- `python -m alpha_suite.backtest --help`

这里的回测定位是研究和 replay 工具，不代表历史结果可以直接外推为实盘结果。
