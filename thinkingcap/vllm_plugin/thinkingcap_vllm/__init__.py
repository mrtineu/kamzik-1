"""vLLM general plugin: registers the ReasoningMLP arms' architecture (out-of-tree model)."""


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen3ReasoningMLPForCausalLM',
                                 'thinkingcap_vllm.model:Qwen3ReasoningMLPForCausalLM')
