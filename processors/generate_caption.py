from transformers import pipeline

caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

def generate_caption(batch):
    batch["caption"] = [caption_pipeline(img)[0]["generated_text"] for img in batch["image"]]
    return batch
