import logging
from transformers import AutoTokenizer
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check():
    model_name = settings.model_name
    logger.info(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check special IDs used in vllm_model_manager.py
    prefix_id = 128259
    suffix_ids = [128009, 128260, 128261, 128257]
    
    logger.info("=== Inspection of IDs used in _format_prompt ===")
    try:
        token = tokenizer.convert_ids_to_tokens([prefix_id])[0]
        logger.info(f"ID {prefix_id} -> {token}")
    except:
        logger.info(f"ID {prefix_id} -> OUT OF RANGE")
        
    for sid in suffix_ids:
        try:
            token = tokenizer.convert_ids_to_tokens([sid])[0]
            logger.info(f"ID {sid} -> {token}")
        except:
            logger.info(f"ID {sid} -> OUT OF RANGE")

    logger.info("\n=== Search for Audio Tokens ===")
    # Search for <custom_token_0> or <|audio|> or similar
    test_tokens = ["<custom_token_0>", "<|audio|>", "<|tara|>", "<|eoa|>"]
    for t in test_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        logger.info(f"Token '{t}' -> ID {tid}")

    # Inspect range around typical offsets
    typical_offsets = [121416, 128266, 128256, 156939]
    for offset in typical_offsets:
        if offset < len(tokenizer):
            try:
                tokens = tokenizer.convert_ids_to_tokens(range(offset, offset + 10))
                logger.info(f"Tokens at offset {offset}: {tokens}")
            except:
                pass

if __name__ == "__main__":
    check()
