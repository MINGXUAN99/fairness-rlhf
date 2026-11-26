"""Formatting and group selection shared by reward training and evaluation."""

GROUP_IDS = {"harmlessness": 0, "helpfulness": 1}


def select_group(dataset, group):
    if group == "pooled":
        return dataset
    group_id = GROUP_IDS[group]
    return dataset.filter(lambda row: row["group_id"] == group_id)


def format_response(tokenizer, prompt, response, mode="auto"):
    use_chat = mode == "chat" or (mode == "auto" and bool(tokenizer.chat_template))
    if use_chat:
        if not tokenizer.chat_template:
            raise ValueError("Chat formatting requested but tokenizer has no chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False,
        ), False
    return f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token or ''}", True


def tokenize_pair(row, tokenizer, max_length, mode="auto"):
    result = {"group_id": row["group_id"]}
    for side in ("chosen", "rejected"):
        text, add_special_tokens = format_response(tokenizer, row["prompt"], row[side], mode)
        result[f"{side}_ids"] = tokenizer(
            text, add_special_tokens=add_special_tokens, truncation=True,
            max_length=max_length,
        )["input_ids"]
    result["usable"] = bool(result["chosen_ids"] and result["rejected_ids"]
                            and result["chosen_ids"] != result["rejected_ids"])
    return result
