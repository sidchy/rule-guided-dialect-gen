# 生成 Prompt -- 用户消息模板

> 每次生成任务时，系统会根据此模板构造用户消息。
> 可用变量：
>   {scene_id} -- 场景标识
>   {example_block} -- 参考例句
>   {core_word} -- 核心词
>   {core_word_def} -- 核心词释义
>   {support_block} -- 辅助词列表
>   {banned_block} -- 禁止词列表
>   {grammar_user_rules} -- 语法约束提示
>   {min_length}, {max_length} -- 句子长度范围

场景：{scene_id}

参考方言例句（注意学习其中的方言风格和用词习惯）：
{example_block}

核心词汇（每句必须使用）：
  - {core_word}（{core_word_def}）

辅助词汇（只有自然时才用，最多用 1 个）：
{support_block}

禁止词汇（即使参考例句出现，也绝对不要写进新句子）：
{banned_block}

请额外遵守这些语法约束：
{grammar_user_rules}

请生成 5 个 {min_length}-{max_length} 字的方言口语长句。
